import torch.nn as nn
import torch
import torch.nn.functional as F


def _init_weights(module):
    if hasattr(module, "weight"):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


def safe_sigmoid(x, threshold=20):
    x = torch.clamp(x, -threshold, threshold)
    return torch.sigmoid(x)


class ObjBoxDiscriminator(nn.Module):
    """
    Discriminator that considers a bounding box and an object class label and judges if its a
    plausible configuration
    """

    def __init__(self, box_dim, obj_dim):
        super(ObjBoxDiscriminator, self).__init__()

        self.obj_dim = obj_dim

        self.D = nn.Sequential(
            nn.Linear(box_dim + obj_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

        self.D.apply(_init_weights)

    def forward(self, objs, boxes, with_grad=False, is_real=False):

        objectCats = to_one_hot_vector(self.obj_dim, objs)

        x = torch.cat([objectCats, boxes], 1)
        reg = None
        if with_grad:
            x.requires_grad = True
            y = self.D(x)
            y = safe_sigmoid(y)
            reg = discriminator_regularizer(y, x, is_real)
            x.requires_grad = False
        else:
            y = self.D(x)
            y = safe_sigmoid(y)
        return y, reg


class ShapeAuxillary(nn.Module):
    """
    Auxiliary discriminator that receives a shape encoding and judges if it is plausible and
    simultaneously predicts a class label for the given shape encoding
    """

    def __init__(self, shape_dim, num_classes):
        super(ShapeAuxillary, self).__init__()

        self.D = nn.Sequential(
            nn.Linear(shape_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Linear(512, num_classes)
        self.discriminator = nn.Linear(512, 1)

        self.D.apply(_init_weights)
        self.classifier.apply(_init_weights)
        self.discriminator.apply(_init_weights)

    def forward(self, shapes):

        backbone = self.D(shapes)
        logits = self.classifier(backbone)
        realfake = torch.sigmoid(self.discriminator(backbone))

        return logits, realfake


class BoxDiscriminator(nn.Module):
    """
    Relationship discriminator based on bounding boxes. For a given object pair, it takes their
    semantic labels, the relationship label and the two bounding boxes of the pair and judges
    whether this is a plausible occurence.
    """

    def __init__(self, box_dim, rel_dim, obj_dim, with_obj_labels=True):
        super(BoxDiscriminator, self).__init__()

        self.rel_dim = rel_dim
        self.obj_dim = obj_dim
        self.with_obj_labels = with_obj_labels
        if self.with_obj_labels:
            in_size = box_dim * 2 + rel_dim + obj_dim * 2
        else:
            in_size = box_dim * 2 + rel_dim

        self.D = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

        self.D.apply(_init_weights)

    def forward(self, objs, triples, boxes, keeps=None, with_grad=False, is_real=False):

        s_idx, predicates, o_idx = triples.chunk(3, dim=1)
        predicates = predicates.squeeze(1)
        s_idx = s_idx.squeeze(1)
        o_idx = o_idx.squeeze(1)
        subjectBox = boxes[s_idx]
        objectBox = boxes[o_idx]
        # for Debug
        assert not torch.isnan(objs).any(), "objs contains nan values"
        assert not torch.isnan(triples).any(), "triples contains nan values"
        if torch.isnan(boxes).any():
            # Replace nan values with -1 using the mask
            boxes = torch.where(
                torch.isnan(boxes), torch.tensor(-1.0, device=boxes.device), boxes
            )

        if keeps is not None:
            assert not torch.isnan(keeps).any(), "keeps contains nan values"
            subjKeeps = keeps[s_idx]
            objKeeps = keeps[o_idx]
            keep_t = ((1 - subjKeeps) + (1 - objKeeps)) > 0

        predicates = to_one_hot_vector(self.rel_dim, predicates)

        if self.with_obj_labels:
            subjectCat = to_one_hot_vector(self.obj_dim, objs[s_idx])
            objectCat = to_one_hot_vector(self.obj_dim, objs[o_idx])

            x = torch.cat([subjectCat, objectCat, predicates, subjectBox, objectBox], 1)

        else:
            x = torch.cat([predicates, subjectBox, objectBox], 1)

        reg = None
        if with_grad:
            if torch.isnan(x).any():
                # Replace nan values with -1 using the mask
                x = torch.where(
                    torch.isnan(x),
                    torch.tensor(-1.0, device=x.device),
                    x,
                )
            x.requires_grad = True
            x.retain_grad()

            y = self.D(x)
            assert not torch.isnan(
                y
            ).any(), "Input y contains nan values before safe_sigmoid"

            y = safe_sigmoid(y)
            reg = discriminator_regularizer(y, x, is_real)
            x.requires_grad = False
        else:
            y = self.D(x)
            y = safe_sigmoid(y)
        if keeps is not None and reg is not None:
            return y[keep_t], reg[keep_t]
        elif keeps is not None and reg is None:
            return y[keep_t], reg
        else:
            return y, reg


def discriminator_regularizer(logits, arg, is_real):

    if torch.isnan(logits).any():
        # Create a mask that identifies nan values in boxes
        nan_mask = torch.isnan(logits)
        # Replace nan values with -1 using the mask
        logits = torch.where(
            torch.isnan(logits), torch.tensor(-1.0, device=logits.device), logits
        )
    assert not torch.isnan(arg).any(), "Input arg contains nan values before backward"

    logits.backward(torch.ones_like(logits), retain_graph=True)
    grad_logits = arg.grad

    if grad_logits is not None:
        assert not torch.isnan(grad_logits).any(), "Gradients of arg contain nan values"
    else:
        print(
            "Warning: grad_logits is None. This may indicate that arg does not contribute to the computation of logits."
        )

    grad_logits_norm = torch.norm(grad_logits, dim=1).unsqueeze(1)
    assert not torch.isnan(grad_logits_norm).any()

    assert grad_logits_norm.shape == logits.shape

    # tf.multiply -> element-wise mul
    if is_real:
        reg = (1.0 - logits) ** 2 * (grad_logits_norm) ** 2
    else:
        reg = (logits) ** 2 * (grad_logits_norm) ** 2

    return reg


def to_one_hot_vector(num_class, label):
    """Converts a label to a one hot vector

    :param num_class: number of object classes
    :param label: integer label values
    :return: a vector of the length num_class containing a 1 at position label, and 0 otherwise
    """
    return torch.nn.functional.one_hot(label, num_class).float()
