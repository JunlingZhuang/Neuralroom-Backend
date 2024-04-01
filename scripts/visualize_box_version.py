import json
import os
import random
import sys
import copy


import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data

# Local application-specific imports
# sys.path.append("../")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from dataset.threedfront_dataset import DatasetSceneGraph
from helpers.util import (
    decode_latent_vector_box,
    params_to_8points_3dfront,
    params_to_8points_no_rot,
)
from helpers.viz_util import (
    create_scene_meshes,
    export_scene_meshes,
    force_room_adjacency,
    render_plotly_sdf,
    rel_to_abs_box_params,
    ROOM_HIER_MAP
)
from model.VAE import VAE
from helpers.visualize_graph import visualize_scene_graph


def prepare_dataset_and_model(args_location = "test/partition_emb_box_250/args.json",ckpt_epoch = 240):
    with open(args_location, "r") as json_file:
        args = json.load(json_file)
    args["device"] = "cpu"
    if os.getcwd().split('\\') [-1]== 'scripts':
        args['exp']='../'+args['exp']
        args['dataset'] = '../'+args['dataset']
        args['data_list'] = '../' + args['data_list']
    device = torch.device(args["device"])

    random.seed(args["manualSeed"])
    torch.manual_seed(args["manualSeed"])

    train_dataset = DatasetSceneGraph(
        root=args["dataset"],
        data_list=args["data_list"],
        split="train",
        shuffle_objs=True,
        use_SDF=args["with_SDF"],
        use_scene_rels=True,
        with_changes=args["with_changes"],
        with_feats=args["with_feats"],
        with_CLIP=args["with_CLIP"],
        large=args["large"],
        seed=False,
        recompute_feats=False,
        recompute_clip=False,
        device=args["device"],
    )

    collate_fn = train_dataset.collate_fn_vaegan_points
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batchSize"],
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=int(2),
    )

    model = VAE(
        root=args["dataset"],
        type=args["network_type"],
        diff_opt="./config/v2_full.yaml",
        vocab=train_dataset.vocab,
        replace_latent=True,
        with_changes=True,
        residual=args["residual"],
        gconv_pooling="avg",
        with_angles=True,
        num_box_params=args["num_box_params"],
        deepsdf=args["with_feats"],
        clip=args["with_CLIP"],
        with_E2=True,
        device=args["device"],
        use_unit_box=args["use_unit_box"],
        gat_heads=args["gat_heads"] if args["network_type"] == "v3_vox" else None,
        num_gat_layers=(
            args["num_gat_layers"] if args["network_type"] == "v3_vox" else None
        ),
    )


    if torch.cuda.is_available():
        model = model.to(device)


    # load model


    model.load_networks(args["exp"], epoch=ckpt_epoch)

    print('model loaded!')
    model.eval()
    model.compute_statistics(
        exp=args["exp"],
        epoch=ckpt_epoch,
        stats_dataloader=dataloader,
        force=False,)
    print('training statistics collected')
    
    bbox_file = os.path.join(args["dataset"], "cat_jid_all.json")
    with open(bbox_file, "r") as read_file:
        box_data = json.load(read_file)

    # TODO: EXPORT UNIT BOX NORMALIZATION PARAM INTO THE TEXT
    box_file = os.path.join(args["dataset"], "boxes_centered_stats_all.txt")

    return args, model, train_dataset, box_data,box_file


def predict_boxes_and_angles(
    is_custom_boundary=False,
    device="cpu",
    random_seed=852,
    model=None,
    unit_box=None,
    args=None,
    data=None,
    unit_box_mean=None,
    unit_box_std=None,
    box_file=None,
):
    """
    if the users do not set custom boundary for the unit, the function will uses original boundary in the training dataset
    input unit_box should be a list
    """
    dec_objs = data["decoder"]["objs"]
    dec_triples = data["decoder"]["triples"]
    dec_text_feat = None
    dec_rel_feat = None
    if args["with_CLIP"]:
        dec_rel_feat = data["decoder"]["rel_feats"]
        dec_text_feat = data["decoder"]["text_feats"]
        dec_rel_feat = dec_rel_feat.to(device)
        dec_text_feat = dec_text_feat.to(device)
    dec_unit_box = data["decoder"]["unit_box"]

    with torch.no_grad():
        np.random.seed(random_seed)

    dec_objs = dec_objs.to(device)
    dec_triples = dec_triples.to(device)
    dec_unit_box = dec_unit_box.to(device)

    if is_custom_boundary and unit_box is not None:
        assert len(unit_box) == 3
        assert unit_box_std is not None
        assert unit_box_mean is not None
        unit_box = (unit_box - unit_box_mean) / unit_box_std
        unit_box = torch.tensor(unit_box).to(device)
        dec_unit_box[:, :] = unit_box

    z = (
        torch.from_numpy(
            np.random.multivariate_normal(
                model.mean_est_box, model.cov_est_box, dec_objs.size(0)
            )
        )
        .float()
        .to(device)
    )

    boxes_pred_den, angles_pred = decode_latent_vector_box(
        z,
        model,
        dec_objs,
        dec_triples,
        dec_text_feat,
        dec_rel_feat,
        dec_unit_box=dec_unit_box,
        args=args,
        box_file=box_file,
    )
    return boxes_pred_den, angles_pred, dec_unit_box


def rationalize_box_params(
    boxes_pred_den,
    angles_pred,
    unit_box_mean,
    unit_box_std,
    dec_unit_box,
    data,
    adj_rel_idx,
):
    """
    rationalize the raw output, forcing adjacent rooms attached and children boxes inside the parent boxes
    return
    - box_points: list of 8 corner points with rotation
    - denormalized_boxes: lisst of box params without rotation (dx,dy,dz,cenx,ceny,cenz)

    """
    angles_pred[-1] = 0.0
    dec_triples = data["decoder"]["triples"]
    obj_to_pidx = data["decoder"]["obj_to_pidx"]
    dec_triples = dec_triples.to(boxes_pred_den.device)
    obj_to_pidx = obj_to_pidx.to(boxes_pred_den.device)

    # unnormalize rel params to abs params
    denormalized_boxes = rel_to_abs_box_params(
        boxes_pred_den,
        obj_to_pidx,
        dec_unit_box[0],
        unit_box_mean,
        unit_box_std,
        angles_pred=angles_pred,
    )

    # force adjacent rooms attach to each other
    adj_room_idxs = torch.where(dec_triples[:, 1] == adj_rel_idx)
    adj_list = dec_triples[adj_room_idxs][:, [0, 2]]
    denormalized_boxes = force_room_adjacency(adj_list, denormalized_boxes, obj_to_pidx)
    box_points_list = []
    box_and_angle_list = []
    for i in range(len(denormalized_boxes)):
        if angles_pred is None:
            box_points_list.append(params_to_8points_no_rot(denormalized_boxes[i]))
        else:
            box_and_angle = np.concatenate(
                [denormalized_boxes[i].float(), angles_pred[i].float()]
            )
            box_and_angle_list.append(box_and_angle)
            box_points_list.append(
                params_to_8points_3dfront(box_and_angle, degrees=True)
            )

    # Concatenate the list of tensors into a single tensor
    box_points = np.array(box_points_list)

    return box_points, denormalized_boxes, angles_pred

def create_data_dic(objs, triples, dataset, unit_box=[6.0, 3.0, 6.0], norm_scale=1):
    data = {}



    # compute pidx
    obj_to_pidx = list(range(len(objs)))
    obj_name2idx = dataset.classes  # {class label : index}
    room_idxs = [obj_name2idx.get(key) for key in list(ROOM_HIER_MAP.keys())]
    for triple in triples:
        sub, pred, obj = triple
        sub_idx = int(objs[int(sub)])
        obj_idx = int(objs[int(obj)])
        fur_idx = None
        room_idx = None
        if sub_idx in room_idxs and obj_idx not in room_idxs:
            room_idx = sub
            fur_idx = obj
        elif sub_idx not in room_idxs and obj_idx in room_idxs:
            room_idx = obj
            fur_idx = sub
        if fur_idx and room_idx:
            obj_to_pidx[fur_idx] = room_idx

    # create grained objs
    objs_grained = []
    for obj in objs:
        label = dataset.classes_r[int(obj)]
        grained_id = dataset.fine_grained_classes[label]
        objs_grained.append(grained_id)

    # add scene node
    scene_idx = len(objs)
    for i, obj in enumerate(objs):
        # check if it is a room node
        if obj_to_pidx[i] == i:
            triples.append([i, 0, scene_idx])

    objs.append(0)
    objs_grained.append(0)
    obj_to_pidx.append(scene_idx)

    # prepare unit_box
    num_objs = len(objs)
    unit_box_mean = dataset.unit_box_mean
    unit_box_std = dataset.unit_box_std
    unit_box = norm_scale * (unit_box - unit_box_mean) / unit_box_std
    unit_box = torch.from_numpy(np.array(unit_box).astype(np.float32))
    unit_box = unit_box.unsqueeze(0).repeat(num_objs, 1)

    # torchify
    data["encoder"] = {}
    data["encoder"]["objs"] = torch.from_numpy(np.array(objs).astype(np.int64))
    data["encoder"]["objs_grained"] = torch.from_numpy(np.array(objs_grained).astype(np.int64))
    data["encoder"]["triples"] = torch.from_numpy(np.array(triples).astype(np.int64))
    data["encoder"]["obj_to_pidx"] = torch.from_numpy(
        np.array(obj_to_pidx).astype(np.int64)
    )
    data["encoder"]["unit_box"] = torch.from_numpy(
        np.array(unit_box).astype(np.float32)
    )
    data["decoder"] = copy.deepcopy(data["encoder"])

    return data

def generate_queried_unit_mesh(
    input_objs = None,
    input_triples = None,
    unit_box=None,
    args=None,
    model=None,
    train_dataset=None,
):
    """
    input nodes, edges, and the custom unit_box(optional), generate the unit mesh
    """
    bbox_file = os.path.join(args["dataset"], "cat_jid_all.json")
    with open(bbox_file, "r") as read_file:
        box_data = json.load(read_file)

    # TODO: EXPORT UNIT BOX NORMALIZATION PARAM INTO THE TEXT
    box_file = os.path.join(args["dataset"], "boxes_centered_stats_all.txt")
    if "unit_box_mean" not in args:
        unit_box_mean = train_dataset.unit_box_mean
        unit_box_std = train_dataset.unit_box_std
    else:
        unit_box_mean = np.array(args["unit_box_mean"])
        unit_box_std = np.array(args["unit_box_std"])
    obj_idx2name = {v: k for k, v in train_dataset.classes.items()}
    rel_idx2name = {k + 1: v for k, v in enumerate(train_dataset.relationships)}
    rel_idx2name[0] = "belong to"
    adj_rel_idx = train_dataset.relationships_dict["adjacent to"]
    device = args['device']

    # parse data
    data = create_data_dic(input_objs,input_triples,train_dataset)
    dec_objs_grained = data["decoder"]["objs_grained"]
    dec_objs = data["decoder"]["objs"]
    dec_triples = data["decoder"]["triples"]
    dec_unit_box = data["decoder"]["unit_box"]
    obj_to_pidx = data["decoder"]["obj_to_pidx"]
    dec_objs, dec_triples, dec_unit_box = (
        dec_objs.to(device),
        dec_triples.to(device),
        dec_unit_box.to(device),
    )

    boxes_pred_den, angles_pred,dec_unit_box = predict_boxes_and_angles(
        is_custom_boundary= unit_box is not None,  # True if custom bbox 
        device="cpu",
        random_seed=852,
        model=model,
        unit_box=unit_box,
        args=args,
        data=data,
        unit_box_mean=unit_box_mean,
        unit_box_std=unit_box_std,
        box_file = box_file
    )
    box_points, denormalized_boxes, angles_pred = rationalize_box_params(
        boxes_pred_den, angles_pred, unit_box_mean, unit_box_std, dec_unit_box, data,adj_rel_idx
    )

    detailed_obj_class = train_dataset.vocab["full_object_idx_to_name_grained"]
    sdf_dir = "DEEPSDF_reconstruction/Meshes"
    # get furniture category
    fur_cat_file = args["dataset"] + "/cat_jid_all.json"
    with open(fur_cat_file, "r") as file:
        fur_cat = json.load(file)
    # trimesh mesh object
    meshes,_,_ = create_scene_meshes(
        dec_objs_grained,
        obj_to_pidx,
        denormalized_boxes,
        angles_pred,
        detailed_obj_class,
        fur_cat,
        sdf_dir,
        retrieve_sdf = False, # export box only meshes
        ceiling_and_floor= False # no ceiling floors
    )
    exp_dir = os.path.join(args['exp'],'mesh')
    mesh_name = '{}.obj'.format(data.get("scan_id", "test"))
    exp_path = os.path.join(exp_dir,mesh_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir,exist_ok = True)
    export_scene_meshes(meshes,dec_objs,obj_idx2name,exp_path)
    return exp_path
    



if __name__ == "__main__":
    
    # inputs
    ARGS_LOCATION = "./test/partition_emb_box_250/args.json"
    QUERY_IDX = 2 # Index in training dataset
    IS_VISUALIZE_GRAPH = False
    IS_VISUALIZE_BOX = False
    IS_EXPORT_MESH = True
    RETRIEVE_FUR_SDF = False
    IS_SEND_MESH_TO_FRONTEND = True

    # setup constants
    args, model, train_dataset, box_data, box_file = prepare_dataset_and_model(
        args_location=ARGS_LOCATION
    )
    if "unit_box_mean" not in args:
        unit_box_mean = train_dataset.unit_box_mean
        unit_box_std = train_dataset.unit_box_std
    else:
        unit_box_mean = np.array(args["unit_box_mean"])
        unit_box_std = np.array(args["unit_box_std"])

    obj_idx2name = {v: k for k, v in train_dataset.classes.items()}
    rel_idx2name = {k + 1: v for k, v in enumerate(train_dataset.relationships)}
    rel_idx2name[0] = "belong to"
    device = args['device']

    # parse data
    data = train_dataset[QUERY_IDX]
    dec_objs_grained = data["decoder"]["objs_grained"]
    dec_objs = data["decoder"]["objs"]
    dec_triples = data["decoder"]["triples"]
    dec_unit_box = data["decoder"]["unit_box"]
    obj_to_pidx = data["decoder"]["obj_to_pidx"]
    dec_objs, dec_triples, dec_unit_box = (
        dec_objs.to(device),
        dec_triples.to(device),
        dec_unit_box.to(device),
    )

    # predict box params
    # custom bounding box will be input of 'unit_box'
    boxes_pred_den, angles_pred,dec_unit_box = predict_boxes_and_angles(
        is_custom_boundary=False,  # True if custom bbox 
        device="cpu",
        random_seed=852,
        model=model,
        unit_box=None,
        args=args,
        data=data,
        unit_box_mean=unit_box_mean,
        unit_box_std=unit_box_std,
        box_file = box_file
    )

    # rationalization
    # box_points: array of 8 corner points
    # denormalized_boxes: array of [size x, size y, size z, center x, center y,center z]
    # angles_pred : array of predicted rotation angles
    box_points, denormalized_boxes, angles_pred = rationalize_box_params(
        boxes_pred_den, angles_pred, unit_box_mean, unit_box_std, dec_unit_box, data
    )

    if IS_VISUALIZE_GRAPH:
        visualize_scene_graph(
            dec_triples.numpy(),
            dec_objs.numpy(),
            obj_idx2name,
            rel_idx2name,
            obj_class_to_color=args["dataset"] + "/class_color.json",
            included_relations=[
                "inside",
                "adjacent to",
            ],
        )
    
    if IS_VISUALIZE_BOX:
        render_plotly_sdf(
        box_points,

        obj_idx2name=obj_idx2name,
        objs=dec_objs,

        shapes_pred=None,

        render_shapes=False,

        render_boxes=True,

        colors=args["dataset"] + "/class_color.json",

        save_as_image=False,

        filename=f"{args['exp']}/scene_render.png",
        obj_to_pidx=obj_to_pidx,

    )
    
    if IS_EXPORT_MESH:
        detailed_obj_class = train_dataset.vocab["full_object_idx_to_name_grained"]
        sdf_dir = "DEEPSDF_reconstruction/Meshes"
        # get furniture category
        fur_cat_file = args["dataset"] + "/cat_jid_all.json"
        with open(fur_cat_file, "r") as file:
            fur_cat = json.load(file)
        # trimesh mesh object
        meshes = create_scene_meshes(
            dec_objs_grained,
            obj_to_pidx,
            denormalized_boxes,
            angles_pred,
            detailed_obj_class,
            fur_cat,
            sdf_dir,
            retrieve_sdf = RETRIEVE_FUR_SDF, # export box only meshes
        )
        exp_dir = os.path.join(args['exp'],'mesh')
        mesh_name = f'{data["scan_id"]}.obj'
        exp_path = os.path.join(exp_dir,mesh_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir,exist_ok = True)
        export_scene_meshes(meshes,dec_objs,obj_idx2name,exp_path)
        print(f"<start>{exp_path}<end>")

    

    



    