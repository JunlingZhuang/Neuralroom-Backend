from __future__ import print_function
import sys

sys.path.append("..")
sys.path.append(".")
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import copy
import json
from helpers.psutil import FreeMemLinux, FreeMem
from helpers.util import normalize_box_params
import random
import pickle
import trimesh

changed_relationships_dict = {
    "left": "right",
    "right": "left",
    "front": "behind",
    "behind": "front",
    "bigger than": "smaller than",
    "smaller than": "bigger than",
    "taller than": "shorter than",
    "shorter than": "taller than",
    "close by": "close by",
    "symmetrical to": "symmetrical to",
    "standing on": "standing on",
    "above": "above",
    "inside": "inside",
    "adjacent to": "adjacent to",
}


def load_ckpt(ckpt):
    map_fn = lambda storage, loc: storage
    if type(ckpt) == str:
        state_dict = torch.load(ckpt, map_location=map_fn)
    else:
        state_dict = ckpt
    return state_dict


# TODO: simplify the categories
# TODO: check if need to change with_feats into with_SDF_feats


# with_feats is for SDF feats
class DatasetSceneGraph(data.Dataset):
    def __init__(
        self,
        root,
        root_3dfront="",
        split="train",
        shuffle_objs=False,
        pass_scan_id=False,
        use_SDF=False,
        use_scene_rels=False,
        data_len=None,
        with_changes=True,
        scale_func="diag",
        eval=False,
        eval_type="addition",
        with_feats=False,
        with_CLIP=False,
        seed=True,
        large=False,
        recompute_feats=False,
        recompute_clip=False,
        class_choice=None,
        data_list=None,
        device="cuda",
    ):
        # Basic setting
        self.seed = seed
        self.with_feats = with_feats  # for SDF
        self.with_CLIP = with_CLIP
        self.cond_model = None
        self.large = large
        self.recompute_feats = recompute_feats
        self.recompute_clip = recompute_clip
        self.scale_func = scale_func
        self.with_changes = with_changes
        self.use_SDF = use_SDF
        self.sdf_res = 64
        self.root = root
        self.class_choice = class_choice
        self.data_list = data_list  # a txt file contains all data filename
        self.scans = []
        self.obj_paths = []
        self.use_scene_rels = use_scene_rels
        self.padding = 0.2
        self.eval = eval
        self.pass_scan_id = pass_scan_id
        self.shuffle_objs = shuffle_objs
        self.files = {}
        self.eval_type = eval_type
        self.root_3dfront = root_3dfront
        self.device = device
        self.unit_box = {}  # a dictionary maps absolute x y z size param
        self.unit_box_mean = None
        self.unit_box_std = None

        if self.root_3dfront == "":
            self.root_3dfront = os.path.join(self.root, "visualization")
            if not os.path.exists(self.root_3dfront):
                os.makedirs(self.root_3dfront)

        # Seed setting for evaluation
        if eval and seed:
            np.random.seed(47)
            torch.manual_seed(47)
            random.seed(47)

        # Memory setting
        if os.name == "nt":  # Windows
            self.fm = FreeMem("GB")
        elif os.name == "posix":  # Linux and other Unix-like OS
            self.fm = FreeMemLinux("GB")
        else:
            raise EnvironmentError("Unsupported Operating System")

        # Vocabulary and Category Settings
        catfile_name = "classes.txt"

        self.catfile = os.path.join(self.root, catfile_name)
        self.cat = {}
        self.vocab = {}
        self._setup_vocab_and_categories(self.class_choice)

        # File List Setup, Split data for Train/Val/Test
        self.data_len = data_len
        self._setup_filelist(self.root, split)

        # Relationship and JSON Data Setup
        self._setup_relationships_and_json()

        # Reduce classes from 151 to selected classes
        self._setup_slected_obj_categories()

        # normalize unit_box
        self.normalize_unit_boxes()

        # Feature Check (if needed)
        self._check_for_missing_features()

    def _setup_vocab_and_categories(self, class_choice):
        # Load full-length relationships and obj categories
        self.relationships = []
        with open(self.catfile, "r") as f:
            self.vocab["full_object_idx_to_name"] = [line.strip() for line in f]
        with open(os.path.join(self.root, "relationships.txt"), "r") as f:
            lines = f.readlines()

        self.vocab["full_rel_idx_to_name"] = [line.strip() for line in lines]
        self.relationships = [line.strip().lower() for line in lines]
        self.relationships_dict = dict(
            zip(self.relationships, range(1, len(self.relationships) + 1))
        )
        self.relationships_dict_r = dict(
            zip(self.relationships_dict.values(), self.relationships_dict.keys())
        )

        # Setup categories
        with open(self.catfile, "r") as f:
            for line in f:
                category = line.rstrip()
                self.cat[category] = category
        if class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))
        self.mapping_full2simple = json.load(
            open(os.path.join(self.root, "mapping.json"), "r")
        )

    def _setup_filelist(self, root_data, split_mode):
        # Read folders in specified dataset location and split into train, val, test splits
        all_file_list = []

        # check if data list txt exists
        if self.data_list is not None:
            with open(self.data_list, "r") as f:
                all_file_list = [line.strip() for line in f]
        else:
            all_file_list = [
                entry
                for entry in os.listdir(root_data)
                if os.path.isdir(os.path.join(root_data, entry))
            ]
        if self.data_len is not None:
            all_file_list = all_file_list[: self.data_len]
        self.filelist = self.split_dataset(
            all_file_list, mode=split_mode, seed=self.seed
        )

    def split_dataset(
        self,
        file_list,
        mode="train",
        train_ratio=0.85,
        val_ratio=0.15,
        seed=None,
    ):
        """
        Splits a list of files into training, validation, and test sets.
        :return: a list contatining either the train, validation, or test file names.
        """
        if train_ratio + val_ratio != 1.0:
            raise ValueError("The sum of the ratios must be 1.")

        if seed is not None:
            random.seed(seed)

        # Shuffle the list for randomness
        shuffled_files = file_list[:]
        random.shuffle(shuffled_files)

        num_files = len(shuffled_files)
        num_train = int(num_files * train_ratio)

        # Split the files into training, validation, and test sets
        split_files = {}
        split_files["train"] = shuffled_files[:num_train]
        split_files["val"] = shuffled_files[num_train:]

        return split_files[mode]

    def _setup_relationships_and_json(self):
        # Setup JSON files for relationships, objects, and bounding boxes
        rel_filename = "relationships_all.json"
        rel_json_file = os.path.join(self.root, rel_filename)
        box_filename = "obj_boxes_all.json"
        box_json_file = os.path.join(self.root, box_filename)
        self.box_normalized_stats = os.path.join(
            self.root, "boxes_centered_stats_all.txt"
        )
        (
            self.relationship_json,
            self.objs_json,
            self.tight_boxes_json,
        ) = self.read_box_and_rel_json(rel_json_file, box_json_file)

        for scene, infos in self.tight_boxes_json.items():
            for id, info in infos.items():
                if "model_path" in info:
                    if info["model_path"]:
                        new_path = info["model_path"].split("/")
                        new_path = os.path.join(*new_path[-4:])
                        info["model_path"] = new_path  # TODO: CHECK ONCE UPDATE GRAPH

    def read_box_and_rel_json(self, json_file, box_json_file):
        """Reads from json files the relationship labels, objects and bounding boxes

        :param json_file: file that stores the objects and relationships
        :param box_json_file: file that stores the oriented 3D bounding box parameters
        :return: three dicts, relationships, objects and boxes
        """
        rel = {}
        objs = {}
        tight_boxes = {}

        with open(box_json_file, "r") as read_file:
            box_data = json.load(read_file)

        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            scan_list = [filename.split("_")[-1] for filename in self.filelist]
            for scan in data["scans"]:
                scan_id = scan["scan"]
                if scan_id in scan_list:
                    relationships = []
                    for relationship in scan["relationships"]:
                        relationship[2]
                        relationships.append(
                            relationship
                        )  # TODO: here minus one and in getitem +1, seems redundant

                    # for every scan in rel json, we append the scan id
                    rel[scan_id] = (
                        relationships  # rel: key is unit id, value is relationship tuple
                    )

                    self.scans.append(scan_id)

                    objects = {}
                    boxes = {}
                    for k, v in scan["objects"].items():
                        # if not self.large:
                        #     objects[int(k)] = self.mapping_full2simple[v]
                        # else:
                        objects[int(k)] = v  # "1": "multi_seat_sofa"
                        if int(k) == 0:  # store the absolute size
                            self.unit_box[scan_id] = box_data[scan_id][k]["param7_abs"][
                                :3
                            ]
                            continue

                        try:
                            boxes[int(k)] = {}
                            parent_id = scan["child_to_parent"].get(k, int(k))
                            boxes[int(k)]["parent"] = parent_id

                            boxes[int(k)]["param7"] = box_data[scan_id][str(k)][
                                "param7"
                            ]
                            # print(boxes[int(k)]["param7"][6])
                            boxes[int(k)]["scale"] = box_data[scan_id][str(k)]["scale"]
                            # print(box_data[scan_id][k]["scale"])
                        except Exception as e:
                            # probably box was not saved because there were 0 points in the instance!
                            print(e)
                            print(f"failed to read box params at {scan_id} box {k}")
                        try:
                            boxes[int(k)]["model_path"] = box_data[scan_id][k][
                                "model_path"
                            ]
                        except Exception as e:
                            print(e)
                            print("no model_path")
                            continue
                    boxes["scene_center"] = box_data[scan_id]["scene_center"]
                    objs[scan_id] = objects
                    tight_boxes[scan_id] = boxes
        return rel, objs, tight_boxes

    def _setup_slected_obj_categories(self):
        # setting up using small or large class categories
        self.vocab["full_object_idx_to_name_grained"] = self.vocab[
            "full_object_idx_to_name"
        ]
        points_classes = list(self.classes.keys())
        if "_unit_" in points_classes:
            points_classes.remove("_unit_")
        self.fine_grained_classes = None
        if not self.large:
            self.fine_grained_classes = dict(
                zip(
                    sorted(
                        [
                            voc.strip("\n")
                            for voc in self.vocab["full_object_idx_to_name"]
                        ]
                    ),
                    range(len(self.vocab["full_object_idx_to_name"])),
                )
            )
            self.vocab["full_object_idx_to_name"] = [
                self.mapping_full2simple[voc.strip("\n")] + "\n"
                for voc in self.vocab["full_object_idx_to_name"]
            ]
            self.classes = dict(
                zip(
                    sorted(
                        list(
                            set(
                                [
                                    voc.strip("\n")
                                    for voc in self.vocab["full_object_idx_to_name"]
                                ]
                            )
                        )
                    ),
                    range(len(list(set(self.vocab["full_object_idx_to_name"])))),
                )
            )
            self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))
            points_classes = list(
                set([self.mapping_full2simple[class_] for class_ in points_classes])
            )

        points_classes_idx = [self.classes[pc] for pc in points_classes]

        self.point_classes_idx = points_classes_idx + [0]
        self.sorted_cat_list = sorted(self.cat)

    def _check_for_missing_features(self):
        # check if all shape features exist. If not they get generated here (once)
        if self.with_feats:
            print(
                "Assume you downloaded the DeepSDF codes and SDFs. If not, please download in README.md"
            )
            # for index in tqdm(range(len(self))):
            #     self.__getitem__(index)
            self.recompute_feats = False

        # check if all clip features exist. If not they get generated here (once)
        if self.with_CLIP:
            self.cond_model, preprocess = clip.load("ViT-B/32", device=self.device)
            if self.device == "cpu":
                self.cond_model_cpu = self.cond_model
            else:
                self.cond_model_cpu, _ = clip.load("ViT-B/32", device="cpu")
            print(f"loading CLIP with cond_model at {self.device}")
            print("Checking for missing clip feats. This can be slow the first time.")
            for index in tqdm(range(len(self))):
                self.__getitem__(index)
            self.recompute_clip = False

    def get_unitbox_mean_and_std(self):
        all_dims = np.array(list(self.unit_box.values()))
        # Calculate mean and standard deviation along the appropriate axis
        self.unit_box_mean = np.mean(all_dims, axis=0)
        self.unit_box_std = np.std(all_dims, axis=0)

    def normalize_unit_boxes(self):
        # Ensure mean and std are calculated
        if self.unit_box_mean is None or self.unit_box_std is None:
            self.get_unitbox_mean_and_std()

        # Normalize each unit box using the calculated mean and standard deviation
        normalized_unit_boxes = {
            scan_id: (dims - self.unit_box_mean) / self.unit_box_std
            for scan_id, dims in self.unit_box.items()
        }

        # Optionally, update the unit_boxes with their normalized dimensions
        self.unit_box = normalized_unit_boxes

        return normalized_unit_boxes

    def norm_points(self, p):
        centroid = np.mean(p, axis=0)
        m = np.max(np.sqrt(np.sum(p**2, axis=1)))
        p = (p - centroid) / float(m)
        return p

    def get_key(self, dict, value):
        for k, v in dict.items():
            if v == value:
                return k
        return None

    def set_clip_path(self, scan_id):
        clip_feats_path = os.path.join(
            self.root_3dfront, scan_id, "CLIP_{}.pkl".format(scan_id)
        )
        if not self.large:
            clip_feats_path = os.path.join(
                self.root_3dfront, scan_id, "CLIP_small_{}.pkl".format(scan_id)
            )
        if not os.path.exists(os.path.join(self.root_3dfront, scan_id)):
            os.makedirs(os.path.join(self.root_3dfront, scan_id))
        if self.recompute_clip:
            clip_feats_path += "tmp"
        return clip_feats_path

    # Load points for debug
    def get_points(self, scan_id):
        if scan_id in self.files:  # Caching
            (points_list, points_norm_list, instances_list) = self.files[scan_id]
        else:
            points_list = np.array([]).reshape(-1, 3)
            points_norm_list = np.array([]).reshape(-1, 3)
            instances_list = np.array([]).reshape(-1, 1)
            for key_, value_ in self.tight_boxes_json[scan_id].items():
                if isinstance(key_, int):
                    path = self.tight_boxes_json[scan_id][key_]["model_path"]
                    # object points
                    if path is not None:
                        raw_mesh = trimesh.load(path)
                        position = self.tight_boxes_json[scan_id][key_]["param7"][3:6]
                        theta = self.tight_boxes_json[scan_id][key_]["param7"][-1]
                        R = np.zeros((3, 3))
                        R[0, 0] = np.cos(theta)
                        R[0, 2] = -np.sin(theta)
                        R[2, 0] = np.sin(theta)
                        R[2, 2] = np.cos(theta)
                        R[1, 1] = 1.0
                        points = raw_mesh.copy().vertices
                        point_norm = self.norm_points(
                            points
                        )  # normliazed in each individual boxes
                        points = points.dot(R) + position  # not centered yet

                    # floor points
                    else:
                        position = self.tight_boxes_json[scan_id][key_]["param7"][3:6]
                        l, w = (
                            self.tight_boxes_json[scan_id][key_]["param7"][0],
                            self.tight_boxes_json[scan_id][key_]["param7"][2],
                        )
                        x = l * np.random.random(1000) + position[0] - l / 2
                        z = w * np.random.random(1000) + position[2] - w / 2
                        y = np.repeat(0, 1000)
                        points = np.vstack((x, y, z)).transpose()
                        point_norm = self.norm_points(points)
                    points_list = np.concatenate((points_list, points), axis=0)
                    points_norm_list = np.concatenate(
                        (points_norm_list, point_norm), axis=0
                    )
                    instances = np.repeat(key_, points.shape[0]).reshape(-1, 1)
                    instances_list = np.concatenate((instances_list, instances), axis=0)

            if self.fm.user_free > 5:
                self.files[scan_id] = (
                    points_list,
                    points_norm_list,
                    instances_list,
                )

        print("shifting points")
        points_list = points_list - np.array(
            self.tight_boxes_json[scan_id]["scene_center"]
        )  # centered in the scene
        return points_list, points_norm_list, instances_list

    def process_instances(self, keys, instance2label, scan_id, key_to_parent):
        """
        Parameters:
        - keys : the whole instances ids, a list of int such as [2, 4, 3, 5, 6, 1, 7]
        - instance2label: a dictionary mapping keys (instance ids) to class label (string), such as {1: 'wardrobe', 2: 'double_bed', 3: 'table', 4: 'cabinet', 5: 'nightstand', 6: 'ceiling_lamp', 7: 'floor'}
        - scan_id: scan_id of current data point, a string that is subset of filename.
        - key_to_parent: a dictionary map key to the parent key
        """
        # key is the scene obj id
        instance2mask = {
            0: 0
        }  # TODO: very strange, in compute_triples, it -1 for every mask, can be simplified
        cat_ids = []
        cat_ids_grained = []
        tight_boxes = []
        counter = 0
        instances_order = []
        selected_shapes = []
        obj_sdf_list = []
        mask_to_parent = [0] * len(keys)
        # process each selected instances in current scan id
        for key in keys:
            if int(key) == 0:
                continue
            # get objects from the selected list of classes of 3dssg
            scene_instance_id = key
            scene_instance_class = instance2label[key]  # a string such as 'wardrobe'
            if not self.large:
                scene_class_id_grained = self.fine_grained_classes[
                    scene_instance_class
                ]  # class id in the entire dataset ids
                scene_instance_class = self.mapping_full2simple[scene_instance_class]
                scene_class_id = self.classes[scene_instance_class]

            else:
                scene_class_id = self.classes[scene_instance_class]

            instance2mask[scene_instance_id] = (
                counter + 1
            )  # map each suffled obj scene id into an ordered scene id
            counter += 1

            # mask to cat:
            if (scene_class_id >= 0) and (scene_instance_id > 0):
                selected_shapes.append(True)
                cat_ids.append(scene_class_id)
                if not self.large:
                    cat_ids_grained.append(scene_class_id_grained)
                else:
                    cat_ids_grained.append(scene_class_id)
                bbox = np.array(self.tight_boxes_json[scan_id][key]["param7"].copy())
                # bbox[3:6] -= np.array(self.tight_boxes_json[scan_id]["scene_center"]) #TODO: already normalized, no need to substract

                instances_order.append(key)
                bins = np.linspace(np.deg2rad(-180), np.deg2rad(180), 24)
                angle = np.digitize(bbox[6], bins)
                bbox = normalize_box_params(bbox, file=self.box_normalized_stats)
                bbox[6] = angle

                tight_boxes.append(bbox)

            if self.use_SDF:
                if self.tight_boxes_json[scan_id][key]["model_path"] is None:
                    obj_sdf_list.append(
                        torch.zeros((1, self.sdf_res, self.sdf_res, self.sdf_res))
                    )  # floor and room
                else:  # TODO: MAKE SURE H5 FILE EXIST
                    model_path = self.tight_boxes_json[scan_id][key]["model_path"]
                    sdf_parent = model_path.replace(
                        "3D-FUTURE-model", "3D-FUTURE-SDF"
                    ).split("\\")
                    sdf_parent = os.path.join(*sdf_parent[:-1])
                    sdf_path = os.path.join(
                        sdf_parent,
                        "ori_sample_grid.h5",  # original furniture model path
                    )
                    h5_f = h5py.File(sdf_path, "r")
                    obj_sdf = h5_f["pc_sdf_sample"][:].astype(np.float32)
                    sdf = torch.Tensor(obj_sdf).view(
                        1, self.sdf_res, self.sdf_res, self.sdf_res
                    )
                    sdf = torch.clamp(sdf, min=-0.2, max=0.2)
                    obj_sdf_list.append(sdf)

            else:
                obj_sdf_list = None

        # create mask to parent list:
        for key in keys:
            if int(key) == 0:
                continue
            mask = instance2mask[key] - 1
            parent_key = key_to_parent[key]
            parent_mask = instance2mask[parent_key] - 1
            mask_to_parent[mask] = parent_mask

        return (
            instances_order,
            instance2mask,
            cat_ids,
            cat_ids_grained,
            tight_boxes,
            obj_sdf_list,
            mask_to_parent,
        )

    def load_clip_feats(self, instances_order):
        """

        : param instances_order: shuffled list of instances id in the scene, such as [6, 4, 3, 2, 1, 5, 8, 7]
        """
        clip_feats_dic = pickle.load(open(self.clip_feats_path, "rb"))

        clip_feats_ins = clip_feats_dic["instance_feats"]  # of shape (ins_num,512)
        clip_feats_order = np.asarray(
            clip_feats_dic["instance_order"]
        )  # of shape (ins_num,)
        ordered_feats = np.empty((0, 512))
        for inst in instances_order:
            index = np.where(clip_feats_order == inst)[0]
            if index.size > 0:  # Ensure the instance is found in clip_feats_order
                # Append the corresponding feature vector to ordered_feats
                ordered_feats = np.vstack([ordered_feats, clip_feats_ins[index]])
        if self.use_scene_rels:
            ordered_feats = np.vstack(
                [ordered_feats, clip_feats_ins[-1][np.newaxis, :]]
            )  # should be the scene's feature
        clip_feats_ins = list(ordered_feats)
        clip_feats_rel = clip_feats_dic["rel_feats"]

        return clip_feats_ins, clip_feats_rel

    def load_sdf_feats(self, instances_order, scan_id, feats_path):
        latents = []
        # for key_, value_ in self.tight_boxes_json[scan_id].items():
        for key_ in instances_order:  # get the objects in order
            if isinstance(key_, int):
                path = self.tight_boxes_json[scan_id][key_]["model_path"]
                if path is None:
                    latent_code = np.zeros(
                        [1, 256]
                    )  # for the floor, latent_code.shape[1]=256
                    # print("why is it none?")
                else:
                    model_id = path.split("/")[-2]
                    latent_code_path = feats_path + model_id + "/sdf.pth"
                    latent_code = torch.load(latent_code_path, map_location="cpu")[0]
                    latent_code = latent_code.detach().numpy()
                latents.append(latent_code)
        latents.append(np.zeros([1, 256]))  # for the room shape
        feats_in = list(np.concatenate(latents, axis=0))

        return feats_in

    def compute_triples(self, scan_id, instance2mask, instance2label):
        triples = []
        words = []
        rel_json = self.relationship_json[scan_id]
        for r in rel_json:  # create relationship triplets from data
            r[0] = int(r[0])
            r[1] = int(r[1])
            r[2] = int(r[2])
            if r[0] in instance2mask.keys() and r[1] in instance2mask.keys():
                subject = instance2mask[r[0]] - 1
                object = instance2mask[r[1]] - 1
                predicate = r[2]
                if subject >= 0 and object >= 0:
                    triples.append([subject, predicate, object])
                    if not self.large:
                        words.append(
                            self.mapping_full2simple[instance2label[r[0]]]
                            + " "
                            + r[3]
                            + " "
                            + self.mapping_full2simple[instance2label[r[1]]]
                        )
                    else:
                        words.append(
                            instance2label[r[0]]
                            + " "
                            + r[3]
                            + " "
                            + instance2label[r[1]]
                        )  # TODO: check
            else:
                continue

        return triples, words

    def add_scene_root_node(
        self,
        triples,
        words,
        cat_ids,
        cat_ids_grained,
        tight_boxes,
        obj_sdf_list,
        mask_to_parent,
        scan_id,
    ):
        scene_idx = len(cat_ids)
        for i, ob in enumerate(cat_ids):
            # check if this obj is a room node
            if mask_to_parent[i] == i:
                triples.append([i, 0, scene_idx])
                words.append(
                    self.get_key(self.classes, ob) + " " + "belong to" + " " + "unit"
                )
        cat_ids.append(0)  # TODO:check
        cat_ids_grained.append(0)
        mask_to_parent.append(scene_idx)

        # unit box
        unit_box = self.unit_box[scan_id]
        unit_box = np.concatenate((unit_box, [0.0, 0.0, 0.0, 0]), axis=0)
        tight_boxes.append(unit_box)
        if self.use_SDF:
            obj_sdf_list.append(
                torch.zeros((1, self.sdf_res, self.sdf_res, self.sdf_res))
            )  # _unit_
        return (
            triples,
            words,
            cat_ids,
            cat_ids_grained,
            tight_boxes,
            obj_sdf_list,
            mask_to_parent,
        )

    def compute_clip_feats(self, cat_ids, instances_order, words):
        num_cat = len(cat_ids) if not self.use_scene_rels else len(cat_ids) - 1
        feats_rel = {}
        obj_cat = []

        with torch.no_grad():
            for i in range(num_cat):
                obj_cat.append(self.get_key(self.classes, cat_ids[i]))
            if (
                self.use_scene_rels
            ):  # only when add root node, we append 'unit' for whole scene
                obj_cat.append("unit")  # TODO:check
            text_obj = clip.tokenize(obj_cat).to(self.device)
            feats_ins = self.cond_model.encode_text(text_obj).detach().cpu().numpy()
            text_rel = clip.tokenize(words).to(self.device)
            rel = self.cond_model.encode_text(text_rel).detach().cpu().numpy()
            for i in range(len(words)):
                feats_rel[words[i]] = rel[i]

        clip_feats_in = {}
        clip_feats_in["instance_feats"] = feats_ins
        clip_feats_in["instance_order"] = instances_order
        clip_feats_in["rel_feats"] = feats_rel
        path = os.path.join(self.clip_feats_path)
        if self.recompute_clip:
            path = path[:-3]

        pickle.dump(clip_feats_in, open(path, "wb"))
        clip_feats_ins = list(clip_feats_in["instance_feats"])
        clip_feats_rel = clip_feats_in["rel_feats"]
        return clip_feats_ins, clip_feats_rel

    # no edge modification
    def prepare_manipulation_output(self, output):
        output["manipulate"] = {}
        if not self.with_changes:
            output["manipulate"]["type"] = "none"

        else:
            if not self.eval:
                if self.with_changes:
                    output["manipulate"]["type"] = ["addition", "none"][
                        np.random.randint(2)
                    ]  # removal is trivial , simplified edge type - so only addition

                else:
                    output["manipulate"]["type"] = "none"
                if output["manipulate"]["type"] == "addition":
                    node_id = self.remove_node_and_relationship(output["encoder"])
                    if node_id >= 0:
                        output["manipulate"]["added"] = node_id
                    else:
                        output["manipulate"]["type"] = "none"
                # elif output["manipulate"]["type"] == "relationship":
                #     rel, pair, suc = self.modify_relship(output["decoder"])
                #     if suc:
                #         output["manipulate"]["relship"] = (rel, pair)
                #     else:
                #         output["manipulate"]["type"] = "none"
            else:
                output["manipulate"]["type"] = self.eval_type
                if output["manipulate"]["type"] == "addition":
                    node_id = self.remove_node_and_relationship(output["encoder"])
                    if node_id >= 0:
                        output["manipulate"]["added"] = node_id
                    else:
                        return
                # elif output["manipulate"]["type"] == "relationship":
                #     rel, pair, suc = self.modify_relship(
                #         output["decoder"], interpretable=True
                #     )
                #     if suc:
                #         output["manipulate"]["relship"] = (rel, pair)
                #     else:
                #         return
        return output

    def __getitem__(self, index):
        scan_id = self.scans[index]

        # instance2label, the whole instance ids in this scene e.g. {1: 'floor', 2: 'wall', 3: 'picture', 4: 'picture'}
        instance2label = self.objs_json[scan_id]
        keys = list(instance2label.keys())  # ids in the scene
        if 0 in keys:  # make sure not include unit right now
            keys.remove(0)
        key_to_parent = {
            key: self.tight_boxes_json[scan_id][key]["parent"] for key in keys
        }
        # a dictionary, key_to_parent[key] = parent key id

        if self.shuffle_objs:
            random.shuffle(keys)

        feats_in = None
        clip_feats_ins = None
        clip_feats_rel = None

        # If true, expected paths to saved clip features will be set here
        if self.with_CLIP:
            self.clip_feats_path = self.set_clip_path(scan_id)

        feats_path = self.root + "/DEEPSDF_reconstruction/Codes/"  # for Graph-to-3D

        # Load points for debug
        # TODO: check if could be deleted
        if self.with_feats and (not os.path.exists(feats_path) or self.recompute_feats):
            points_list, points_norm_list, instances_list = self.get_points(scan_id)

        # process and map each scene instance ids
        (
            instances_order,
            instance2mask,
            cat_ids,
            cat_ids_grained,
            tight_boxes,
            obj_sdf_list,
            mask_to_parent,
        ) = self.process_instances(keys, instance2label, scan_id, key_to_parent)

        if self.with_CLIP and os.path.exists(self.clip_feats_path):
            # If precomputed features exist, we simply load them
            clip_feats_ins, clip_feats_rel = self.load_clip_feats(instances_order)

        if self.with_feats:
            # If precomputed sdf latent codes exist, we simply load them
            feats_in = self.load_sdf_feats(instances_order, scan_id, feats_path)

        # compute triples and CLIP words
        # triple use idx of obj list to refer to obj
        triples, words = self.compute_triples(scan_id, instance2mask, instance2label)

        if self.use_scene_rels:
            # add _scene_ object and _in_scene_ connections
            (
                triples,
                words,
                cat_ids,
                cat_ids_grained,
                tight_boxes,
                obj_sdf_list,
                mask_to_parent,
            ) = self.add_scene_root_node(
                triples,
                words,
                cat_ids,
                cat_ids_grained,
                tight_boxes,
                obj_sdf_list,
                mask_to_parent,
                scan_id,
            )
        # if features are requested but the files don't exist, we run all loaded pointclouds through clip
        # to compute them and then save them for future usage
        if (
            self.with_CLIP
            and (not os.path.exists(self.clip_feats_path) or clip_feats_ins is None)
            and self.cond_model is not None
        ):
            clip_feats_ins, clip_feats_rel = self.compute_clip_feats(
                cat_ids, instances_order, words
            )

        # prepare outputs
        # objs are a list of true class idx that in current split scene
        # triples are (objs idx + true relationship id + objs idx) tuple

        # prepare unit_box
        unit_box = self.unit_box[scan_id]
        num_objs = len(cat_ids)
        unit_box = torch.from_numpy(np.array(unit_box).astype(np.float32))
        unit_box = unit_box.unsqueeze(0).repeat(num_objs, 1)

        output = {}
        output["encoder"] = {}
        output["encoder"]["objs"] = torch.from_numpy(np.array(cat_ids).astype(np.int64))
        output["encoder"]["objs_grained"] = torch.from_numpy(
            np.array(cat_ids_grained).astype(np.int64)
        )  # not needed for encoder
        output["encoder"]["triples"] = torch.from_numpy(
            np.array(triples).astype(np.int64)
        )
        output["encoder"]["boxes"] = torch.from_numpy(
            np.array(tight_boxes).astype(np.float32)
        )
        output["encoder"]["words"] = words
        output["encoder"]["obj_to_pidx"] = torch.from_numpy(
            np.array(mask_to_parent).astype(np.int64)
        )
        output["encoder"]["unit_box"] = torch.from_numpy(
            np.array(unit_box).astype(np.float32)
        )

        if self.with_CLIP:
            output["encoder"]["text_feats"] = torch.from_numpy(
                np.array(clip_feats_ins).astype(np.float32)
            )
            clip_feats_rel_new = []
            if clip_feats_rel is not None:
                for word in words:
                    clip_feats_rel_new.append(clip_feats_rel[word])
                output["encoder"]["rel_feats"] = torch.from_numpy(
                    np.array(clip_feats_rel_new).astype(np.float32)
                )

        if self.with_feats:
            output["encoder"]["feats"] = torch.from_numpy(
                np.array(feats_in).astype(np.float32)
            )

        output["decoder"] = copy.deepcopy(output["encoder"])
        output = self.prepare_manipulation_output(output)
        output["scan_id"] = scan_id
        output["instance_id"] = instances_order

        return output

    def remove_node_and_relationship(self, graph):
        """Automatic random removal of certain nodes at training time to enable training with changes. In that case
        also the connecting relationships of that node are removed

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :return: index of the removed node
        """

        node_id = -1
        # dont remove layout components, like floor. those are essential
        # TODO: need to modify when consider room nodes
        excluded_classes = [
            "floor",
            "bedroom",
            "diningroom",
            "livingroom",
            "wall",
        ]
        excluded = [
            self.classes[cls] for cls in excluded_classes if cls in self.classes
        ]

        trials = 0
        while node_id < 0 or graph["objs"][node_id] in excluded:
            if trials > 100:
                return -1
            trials += 1
            node_id = np.random.randint(len(graph["objs"]) - 1)

        mask = torch.ones(len(graph["objs"]), dtype=torch.bool)
        mask[node_id] = False
        graph["objs"] = graph["objs"][mask]
        graph["objs_grained"] = graph["objs_grained"][mask]
        if self.with_feats:
            graph["feats"] = graph["feats"][mask]
        if self.with_CLIP:
            graph["text_feats"] = graph["text_feats"][mask]

        graph["boxes"] = graph["boxes"][mask]
        graph["unit_box"] = graph["unit_box"][mask]
        graph["obj_to_pidx"] = graph["obj_to_pidx"][mask]

        to_rm_indices = []
        mask_rel = torch.ones(len(graph["triples"]), dtype=torch.bool)
        for i, (sub, pred, obj) in enumerate(graph["triples"]):
            if sub == node_id or obj == node_id:
                to_rm_indices.append(i)
                mask_rel[i] = False

        for i in reversed(
            to_rm_indices
        ):  # Remove in reverse order to maintain correct indexing
            graph["words"].pop(i)

        graph["triples"] = graph["triples"][mask_rel]
        if self.with_CLIP:
            graph["rel_feats"] = graph["rel_feats"][mask_rel]

        for i in range(len(graph["triples"])):
            if graph["triples"][i][0] > node_id:
                graph["triples"][i][0] -= 1

            if graph["triples"][i][2] > node_id:
                graph["triples"][i][2] -= 1

        return node_id

    def modify_relship(
        self, graph, interpretable=False
    ):  # TODO: if modify relationship , here need to modify too
        """Change a relationship type in a graph

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :param interpretable: boolean, if true choose a subset of easy to interpret relations for the changes
        :return: index of changed triplet, a tuple of affected subject & object, and a boolean indicating if change happened
        """

        # rels 26 -> 0
        """15 same material as' '14 same super category as' '13 same style as' '12 symmetrical to' '11 shorter than' '10 taller than' '9 smaller than'
         '8 bigger than' '7 standing on' '6 above' '5 close by' '4 behind' '3 front' '2 right' '1 left'
         '0: none"""
        # subset of edge labels that are spatially interpretable (evaluatable via geometric contraints)
        interpretable_rels = [1, 2, 3, 4, 8, 9, 10, 11]
        rel_dict = {
            1: "left",
            2: "right",
            3: "front",
            4: "behind",
            8: "bigger than",
            9: "smaller than",
            10: "taller than",
            11: "shorter than",
        }
        did_change = False
        trials = 0
        excluded_classes = [
            "floor",
            "bathroom",
            "bedroom",
            "diningroom",
            "livingroom",
            "kitchen",
            "wall",
        ]
        excluded = [
            self.classes[cls] for cls in excluded_classes if cls in self.classes
        ]
        eval_excluded = excluded

        while not did_change and trials < 1000:
            idx = np.random.randint(len(graph["triples"]))
            sub, pred, obj = graph["triples"][idx].tolist()
            sub = int(sub)
            pred = int(pred)
            obj = int(obj)
            trials += 1

            if pred == 0:
                continue
            if graph["objs"][obj] in excluded or graph["objs"][sub] in excluded:
                continue
            if interpretable:
                if (
                    graph["objs"][obj] in eval_excluded
                    or graph["objs"][sub] in eval_excluded
                ):  # don't use the floor
                    continue
                if pred not in interpretable_rels:
                    continue
                else:
                    new_pred = self.relationships_dict[
                        changed_relationships_dict[self.relationships_dict_r[pred]]
                    ]
            else:
                # close by, above, standing on, same material......
                if (
                    self.relationships_dict_r[pred]
                    == changed_relationships_dict[self.relationships_dict_r[pred]]
                ):
                    new_pred = np.random.randint(1, len(self.relationships))
                    # didnt change
                    if new_pred == pred:
                        continue
                # left, right, front, behind, bigger, smaller.....
                else:
                    new_pred = self.relationships_dict[
                        changed_relationships_dict[self.relationships_dict_r[pred]]
                    ]

            graph["words"][idx] = graph["words"][idx].replace(
                self.relationships_dict_r[int(graph["triples"][idx][1])],
                self.relationships_dict_r[new_pred],
            )
            graph["changed_id"] = idx
            updated_row = torch.tensor(
                [sub, new_pred, obj], dtype=graph["triples"].dtype
            )
            graph["triples"][idx] = updated_row

            did_change = True
        return idx, (sub, obj), did_change

    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else:
            return len(self.scans)

    def collate_fn_vaegan(self, batch, use_points=False):
        """
        Collate function to be used when wrapping a RIODatasetSceneGraph in a
        DataLoader. Returns a dictionary
        """

        out = {}

        out["scene_points"] = []
        out["scan_id"] = []
        out["instance_id"] = []

        out["missing_nodes"] = []
        out["missing_nodes_decoder"] = []
        out["manipulated_nodes"] = []
        global_node_id = 0
        global_dec_id = 0
        for i in range(len(batch)):
            if batch[i] == -1:
                return -1
            # notice only works with single batches
            out["scan_id"].append(batch[i]["scan_id"])
            out["instance_id"].append(batch[i]["instance_id"])

            if batch[i]["manipulate"]["type"] == "addition":
                out["missing_nodes"].append(
                    global_node_id + batch[i]["manipulate"]["added"]
                )
                out["missing_nodes_decoder"].append(
                    global_dec_id + batch[i]["manipulate"]["added"]
                )
            elif batch[i]["manipulate"]["type"] == "relationship":
                rel, (sub, obj) = batch[i]["manipulate"]["relship"]
                out["manipulated_nodes"].append(global_dec_id + sub)
                out["manipulated_nodes"].append(global_dec_id + obj)

            if "scene" in batch[i]:
                out["scene_points"].append(batch[i]["scene"])

            global_node_id += len(batch[i]["encoder"]["objs"])
            global_dec_id += len(batch[i]["decoder"]["objs"])

        for key in ["encoder", "decoder"]:
            all_objs, all_boxes, all_triples = [], [], []
            all_objs_grained = []
            all_obj_to_scene, all_triple_to_scene = [], []
            all_points = []
            all_sdfs = []
            all_feats = []
            all_text_feats = []
            all_rel_feats = []
            all_obj_to_pidx = []
            all_unit_box = []

            obj_offset = 0

            for i in range(len(batch)):
                if batch[i] == -1:
                    print("this should not happen")
                    continue
                (objs, triples, boxes) = (
                    batch[i][key]["objs"],
                    batch[i][key]["triples"],
                    batch[i][key]["boxes"],
                )
                obj_to_pidx = batch[i][key]["obj_to_pidx"]
                unit_box = batch[i][key]["unit_box"]

                if "points" in batch[i][key]:
                    all_points.append(batch[i][key]["points"])
                if "sdfs" in batch[i][key]:
                    all_sdfs.append(batch[i][key]["sdfs"])
                if "feats" in batch[i][key]:
                    all_feats.append(batch[i][key]["feats"])
                if "text_feats" in batch[i][key]:
                    all_text_feats.append(batch[i][key]["text_feats"])
                if "rel_feats" in batch[i][key]:
                    if "changed_id" in batch[i][key]:
                        idx = batch[i][key]["changed_id"]
                        if self.with_CLIP:
                            text_rel = clip.tokenize(batch[i][key]["words"][idx]).to(
                                "cpu"
                            )
                            rel = (
                                self.cond_model_cpu.encode_text(text_rel)
                                .detach()
                                .numpy()
                            )
                            batch[i][key]["rel_feats"][idx] = torch.from_numpy(
                                np.squeeze(rel)
                            )

                    all_rel_feats.append(batch[i][key]["rel_feats"])

                num_objs, num_triples = objs.size(0), triples.size(0)

                all_objs.append(batch[i][key]["objs"])
                all_objs_grained.append(batch[i][key]["objs_grained"])
                all_boxes.append(boxes)
                all_obj_to_pidx.append(obj_to_pidx)
                all_unit_box.append(unit_box)

                if triples.dim() > 1:
                    triples = triples.clone()
                    triples[:, 0] += obj_offset
                    triples[:, 2] += obj_offset

                    all_triples.append(triples)
                    all_triple_to_scene.append(torch.LongTensor(num_triples).fill_(i))

                all_obj_to_scene.append(torch.LongTensor(num_objs).fill_(i))

                obj_offset += num_objs

            all_objs = torch.cat(all_objs)
            all_objs_grained = torch.cat(all_objs_grained)
            all_boxes = torch.cat(all_boxes)
            all_obj_to_pidx = torch.cat(all_obj_to_pidx)
            all_unit_box = torch.cat(all_unit_box)

            all_obj_to_scene = torch.cat(all_obj_to_scene)

            if len(all_triples) > 0:
                all_triples = torch.cat(all_triples)
                all_triple_to_scene = torch.cat(all_triple_to_scene)
            else:
                return -1

            outputs = {
                "objs": all_objs,
                "objs_grained": all_objs_grained,
                "triples": all_triples,
                "boxes": all_boxes,
                "obj_to_scene": all_obj_to_scene,
                "triple_to_scene": all_triple_to_scene,
                "obj_to_pidx": all_obj_to_pidx,
                "unit_box": all_unit_box,
            }

            if len(all_sdfs) > 0:
                outputs["sdfs"] = torch.cat(all_sdfs)
            if len(all_points) > 0:
                all_points = torch.cat(all_points)
                outputs["points"] = all_points

            if len(all_feats) > 0:
                all_feats = torch.cat(all_feats)
                outputs["feats"] = all_feats
            if len(all_text_feats) > 0:
                all_text_feats = torch.cat(all_text_feats)
                outputs["text_feats"] = all_text_feats
            if len(all_rel_feats) > 0:
                all_rel_feats = torch.cat(all_rel_feats)
                outputs["rel_feats"] = all_rel_feats
            out[key] = outputs

        return out

    def collate_fn_vaegan_points(self, batch):
        """Wrapper of the function collate_fn_vaegan to make it also return points"""
        return self.collate_fn_vaegan(batch, use_points=True)


if __name__ == "__main__":
    dataset = DatasetSceneGraph(
        root="/media/ymxlzgy/Data/Dataset/3D-FRONT",
        split="val_scans",
        shuffle_objs=True,
        use_SDF=False,
        use_scene_rels=True,
        with_changes=True,
        with_feats=False,
        with_CLIP=True,
        large=False,
        seed=False,
        recompute_clip=False,
    )
    a = dataset[0]

    for x in ["encoder", "decoder"]:
        en_obj = a[x]["objs"].cpu().numpy().astype(np.int32)
        en_triples = a[x]["triples"].cpu().numpy().astype(np.int32)
        # instance
        sub = en_triples[:, 0]
        obj = en_triples[:, 2]
        # cat
        instance_ids = np.array(sorted(list(set(sub.tolist() + obj.tolist()))))  # 0-n
        cat_ids = en_obj[instance_ids]
        texts = [dataset.classes_r[cat_id] for cat_id in cat_ids]
        objs = dict(zip(instance_ids.tolist(), texts))
        objs = {str(key): value for key, value in objs.items()}
        for rel in en_triples[:, 1]:
            if rel == 0:
                txt = "in"
                txt_list.append(txt)
                continue
            txt = dataset.relationships_dict_r[rel]
            txt_list.append(txt)
        txt_list = np.array(txt_list)
        rel_list = np.vstack((sub, obj, en_triples[:, 1], txt_list)).transpose()
        print(a["scan_id"])
