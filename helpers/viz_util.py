import json
import yaml
import trimesh
import os
import random
import open3d as o3d
import numpy as np
from helpers.util import (
    fit_shapes_to_box_3dfront,
    params_to_8points_no_rot,
)
import json
import torch
import plotly.graph_objects as go


def load_semantic_scene_graphs_custom(
    yml_relationships, color_palette, rel_label_to_id, with_manipuation=False
):
    scene_graphs = {}

    graphs = yaml.load(open(yml_relationships, "r"))
    for scene_id, scene in graphs["Scenes"].items():

        scene_graphs[str(scene_id)] = {}
        scene_graphs[str(scene_id)]["objects"] = []
        scene_graphs[str(scene_id)]["relationships"] = []
        scene_graphs[str(scene_id)]["node_mask"] = [1] * len(scene["nodes"])
        scene_graphs[str(scene_id)]["edge_mask"] = [1] * len(scene["relships"])

        for i, n in enumerate(scene["nodes"]):
            obj_item = {
                "ply_color": color_palette[i % len(color_palette)],
                "id": str(i),
                "label": n,
            }
            scene_graphs[str(scene_id)]["objects"].append(obj_item)
        for r in scene["relships"]:
            rel_4 = [r[0], r[1], rel_label_to_id[r[2]], r[2]]
            scene_graphs[str(scene_id)]["relationships"].append(rel_4)
        counter = len(scene["nodes"])
        if with_manipuation:
            for m in scene["manipulations"]:
                if m[1] == "add":
                    # visualize an addition
                    # ['chair', 'add', [[2, 'standing on'], [1, 'left']]]
                    obj_item = {
                        "ply_color": color_palette[counter % len(color_palette)],
                        "id": str(counter),
                        "label": m[0],
                    }
                    scene_graphs[str(scene_id)]["objects"].append(obj_item)

                    scene_graphs[str(scene_id)]["node_mask"].append(0)
                    for mani_rel in m[2]:
                        rel_4 = [
                            counter,
                            mani_rel[0],
                            rel_label_to_id[mani_rel[1]],
                            mani_rel[1],
                        ]
                        scene_graphs[str(scene_id)]["relationships"].append(rel_4)
                        scene_graphs[str(scene_id)]["edge_mask"].append(0)
                    counter += 1
                if m[1] == "rel":
                    # visualize changes in the relationship
                    for rid, r in enumerate(
                        scene_graphs[str(scene_id)]["relationships"]
                    ):
                        s, o, p, l = r
                        if isinstance(m[2][3], list):
                            # ['', 'rel', [0, 1, 'right', [0, 1, 'left']]]
                            if (
                                s == m[2][0]
                                and o == m[2][1]
                                and l == m[2][2]
                                and s == m[2][3][0]
                                and o == m[2][3][1]
                            ):
                                # a change on the SAME (s, o) pair, indicate the change
                                scene_graphs[str(scene_id)]["edge_mask"][rid] = 0
                                scene_graphs[str(scene_id)]["relationships"][rid][3] = (
                                    m[2][2] + "->" + m[2][3][2]
                                )
                                scene_graphs[str(scene_id)]["relationships"][rid][2] = (
                                    rel_label_to_id[m[2][3][2]]
                                )
                                break
                            elif s == m[2][0] and o == m[2][1] and l == m[2][2]:
                                # overwrite this edge with a new pair (s,o)
                                del scene_graphs[str(scene_id)]["edge_mask"][rid]
                                del scene_graphs[str(scene_id)]["relationships"][rid]
                                scene_graphs[str(scene_id)]["edge_mask"].append(0)
                                new_edge = [
                                    m[2][3][0],
                                    m[2][3][1],
                                    rel_label_to_id[m[2][3][2]],
                                    m[2][3][2],
                                ]
                                scene_graphs[str(scene_id)]["relationships"].append(
                                    new_edge
                                )
                        else:
                            # ['', 'rel', [0, 1, 'right', 'left']]
                            if s == m[2][0] and o == m[2][1] and l == m[2][2]:
                                scene_graphs[str(scene_id)]["edge_mask"][rid] = 0
                                scene_graphs[str(scene_id)]["relationships"][rid][3] = (
                                    m[2][2] + "->" + m[2][3]
                                )
                                scene_graphs[str(scene_id)]["relationships"][rid][2] = (
                                    rel_label_to_id[m[2][3]]
                                )
                                break

    return scene_graphs


def load_semantic_scene_graphs(json_relationships, json_objects):
    scene_graphs_obj = {}

    with open(json_objects, "r") as read_file:
        data = json.load(read_file)
        for s in data["scans"]:
            scan = s["scan"]
            objs = s["objects"]
            scene_graphs_obj[scan] = {}
            scene_graphs_obj[scan]["scan"] = scan
            scene_graphs_obj[scan]["objects"] = []
            for obj in objs:
                scene_graphs_obj[scan]["objects"].append(obj)
    scene_graphs = {}
    with open(json_relationships, "r") as read_file:
        data = json.load(read_file)
        for s in data["scans"]:
            scan = s["scan"]
            split = str(s["split"])
            if scan + "_" + split not in scene_graphs:
                scene_graphs[scan + "_" + split] = {}
                scene_graphs[scan + "_" + split]["objects"] = []
                print("WARNING: no objects for this scene")
            scene_graphs[scan + "_" + split]["relationships"] = []
            for k in s["objects"].keys():
                ob = s["objects"][k]
                for i, o in enumerate(scene_graphs_obj[scan]["objects"]):
                    if o["id"] == k:
                        inst = i
                        break
                scene_graphs[scan + "_" + split]["objects"].append(
                    scene_graphs_obj[scan]["objects"][inst]
                )
            for rel in s["relationships"]:
                scene_graphs[scan + "_" + split]["relationships"].append(rel)
    return scene_graphs


def read_relationships(read_file):
    relationships = []
    with open(read_file, "r") as f:
        for line in f:
            relationship = line.rstrip().lower()
            relationships.append(relationship)
    return relationships


# from kaleido.scopes.plotly import PlotlyScope
def calculate_distance(box1, box2):
    """
    Calculate the minimum distance between two boxes along x and z directions.
    Each box is represented by an array: [size_x, size_y, size_z, center_x, center_y, center_z].
    """
    size1, center1 = box1[:3], box1[3:]
    size2, center2 = box2[:3], box2[3:]

    distance_x = abs(center1[0] - center2[0]) - (size1[0] + size2[0]) / 2
    distance_z = abs(center1[2] - center2[2]) - (size1[2] + size2[2]) / 2

    return max(distance_x, 0), max(distance_z, 0)


def force_room_pair_attach(room1_idx, room2_idx, boxes, obj_to_pidx):
    """
    Adjust the position of box1 to eliminate any gap with box2 along the x or z direction.
    Moves box1 and its associated furniture along the direction of minimum non-zero distance.
    """
    all_boxes = boxes.clone()
    room1 = all_boxes[room1_idx]
    room2 = all_boxes[room2_idx]
    distance_x, distance_z = calculate_distance(room1, room2)

    # Determine direction to move based on minimum non-zero distance
    move_x = move_z = 0
    if distance_x > 0 and (distance_x <= distance_z or distance_z == 0):
        # Move along x
        direction = torch.sign(room2[3] - room1[3])
        move_x = direction * distance_x
    if distance_z > 0 and (distance_z < distance_x or distance_x == 0):
        # Move along z
        direction = torch.sign(room2[5] - room1[5])
        move_z = direction * distance_z

    for i, room in enumerate(all_boxes):
        if obj_to_pidx[i] == room1_idx:  # move room1 and associated furniture
            room[3] += move_x
            room[5] += move_z

    return all_boxes


def force_room_adjacency(adjacency_list, boxes, obj_to_pidx):
    """
    Check and adjust positions of all adjacent room pairs iteratively until no more adjustments are needed.
    """
    adjusted = True
    adjusted_boxes = boxes.clone()
    while adjusted:
        adjusted = False
        for pair in adjacency_list:
            room1_idx, room2_idx = pair
            before_distance = calculate_distance(
                adjusted_boxes[room1_idx], adjusted_boxes[room2_idx]
            )
            adjusted_boxes = force_room_pair_attach(
                room1_idx, room2_idx, adjusted_boxes, obj_to_pidx
            )
            after_distance = calculate_distance(
                adjusted_boxes[room1_idx], adjusted_boxes[room2_idx]
            )
            if before_distance != after_distance:
                adjusted = True
    return adjusted_boxes


def params_to_8points_3dfront(box, degrees=False):
    """Given bounding box as 7 parameters: l, h, w, cx, cy, cz, z, compute the 8 corners of the box"""
    l, h, w, px, py, pz, angle = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([l.item() / 2 * i, h.item() / 2 * j, w.item() / 2 * k])
    points = np.asarray(points)
    points = points.dot(get_rotation_3dfront(angle.item(), degree=degrees))
    points += np.expand_dims(np.array([px.item(), py.item(), pz.item()]), 0)
    return points


def get_rotation_3dfront(y, degree=True):
    if degree:
        y = np.deg2rad(y)
    rot = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
    return rot


def rel_to_abs_box_params(
    boxes, obj_to_pidx, unit_box_size, unit_mean, unit_std, angles_pred=None
):
    """Convert the relative params (dx,dy,dz,origin_x,origin_y,origin_z), which all
    normalized within their parent bbox, into abs params  (dx,dy,dz,cen_x,cen_y,cen_z)
    """
    # Denormalize unit_box to abs size
    unit_box_size = unit_box_size * unit_std + unit_mean
    # Calculate unit box's origin based on its center being at (0,0,0)
    unit_origin = torch.tensor([0.0, 0.0, 0.0], device=unit_box_size.device) - (
        unit_box_size / 2
    )
    # Initialize tensor for unnormalized boxes
    unnormalized_boxes = torch.zeros_like(boxes)
    num_objs = boxes.size(0)
    # Unnormalize room boxes using unit box size
    for i in range(num_objs - 1):  # Exclude the unit itself
        if obj_to_pidx[i] == i:  # This is a room
            norm_box = boxes[i]
            norm_box = torch.clamp(norm_box, min=0.0, max=1.0)
            size = norm_box[:3] * unit_box_size
            for j in range(3):
                max_val = 1.0 - norm_box[j]
                norm_box[j + 3] = torch.clamp(norm_box[j + 3], min=0.0, max=max_val)
            origin = norm_box[3:] * unit_box_size + unit_origin
            # force same origin y
            origin[1] = unit_origin[1]
            cen = origin + (size / 2)
            unnormalized_boxes[i, :3] = size
            unnormalized_boxes[i, 3:] = cen

    # Unnormalize furniture boxes using their parent room boxes
    for i in range(num_objs - 1):  # Exclude the unit itself
        if obj_to_pidx[i] != i:  # This is furniture
            parent_idx = obj_to_pidx[i]
            parent_size = unnormalized_boxes[parent_idx, :3]
            parent_origin = unnormalized_boxes[parent_idx, 3:] - (parent_size / 2)
            norm_box = boxes[i]
            norm_box = torch.clamp(norm_box, min=0.0, max=1.0)
            size = norm_box[:3] * parent_size
            if angles_pred is not None:
                fur_angle = angles_pred[i] % 360
                if fur_angle not in (0.0, 180.0, -180.0):
                    for j in range(3):
                        if fur_angle not in (90.0, -90.0, 270.0, -270):
                            max_val = min(1.0 - norm_box[j], 1.0 - norm_box[2 - j])
                        else:
                            max_val = 1.0 - norm_box[2 - j]
                        max_val = max(max_val, 0.0)
                        norm_box[j + 3] = torch.clamp(
                            norm_box[j + 3], min=0.0, max=max_val
                        )
                else:
                    for j in range(3):
                        max_val = max(1.0 - norm_box[j], 0.0)
                        norm_box[j + 3] = torch.clamp(
                            norm_box[j + 3], min=0.0, max=max_val
                        )
            else:
                print("input should contain angles")
                raise ValueError
            origin = norm_box[3:] * parent_size + parent_origin
            # force same origin y
            origin[1] = parent_origin[1]
            cen = origin + (size / 2)
            unnormalized_boxes[i, :3] = size
            unnormalized_boxes[i, 3:] = cen

    # Handle unit box (assumed to be the last entry)
    unnormalized_boxes[-1, :3] = unit_box_size
    unnormalized_boxes[-1, 3:] = unit_origin + (unit_box_size / 2)

    return unnormalized_boxes


def rationalize_bbox(
    unnormalized_box_pts, obj_to_pidx, unit_center=torch.tensor([0.0, 0.0, 0.0])
):
    pass


def adjust_positions_to_center(
    unnormalized_boxes, obj_to_pidx, unit_center=torch.tensor([0.0, 0.0, 0.0])
):
    num_objs = unnormalized_boxes.size(0)

    # Calculate the centroid of all rooms and adjust their positions
    room_indices = [
        i for i in range(num_objs - 1) if obj_to_pidx[i] == i
    ]  # Identify rooms
    if room_indices:
        room_centers = unnormalized_boxes[room_indices, 3:]
        rooms_centroid = room_centers.mean(0)
        rooms_offset = unit_center - rooms_centroid
        for i in room_indices:
            unnormalized_boxes[i, 3:] += rooms_offset

    # Adjust furniture positions relative to the adjusted room positions
    for room_idx in room_indices:
        furniture_indices = [
            i
            for i in range(num_objs - 1)
            if obj_to_pidx[i] == room_idx and obj_to_pidx[i] != i
        ]
        if furniture_indices:
            # Use the room center AFTER adjustment as the target for furniture centroid
            target_center = unnormalized_boxes[room_idx, 3:]
            furniture_centers = unnormalized_boxes[furniture_indices, 3:]
            furniture_centroid = furniture_centers.mean(0)
            furniture_offset = target_center - furniture_centroid
            for fur_idx in furniture_indices:
                unnormalized_boxes[fur_idx, 3:] += furniture_offset

    return unnormalized_boxes


def box_vertices_faces(cornerpoints):
    vertices = cornerpoints
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 7],
            [1, 7, 3],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [4, 6, 7],
            [4, 7, 5],
        ]
    )
    return vertices, faces


def make_room_mesh(predBox, predAngle):
    box_and_angle = torch.cat([predBox.float(), predAngle.float()])
    box_points = params_to_8points_3dfront(box_and_angle, degrees=True)
    vertices = box_points
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 7],
            [1, 7, 3],
            # [2, 3, 7],
            # [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [4, 6, 7],
            [4, 7, 5],
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def create_floor(box_and_angle):
    points_list_x = []
    points_list_z = []
    box_points = params_to_8points_3dfront(box_and_angle, degrees=True)
    vertices, faces = box_vertices_faces(box_points)
    # points_list_x.append(box_points[0:2, 0])
    # points_list_x.append(box_points[4:6, 0])
    # points_list_z.append(box_points[0:2, 2])
    # points_list_z.append(box_points[4:6, 2])
    # points_x = np.array(points_list_x).reshape(-1, 1)
    # points_y = np.zeros(points_x.shape)
    # points_z = np.array(points_list_z).reshape(-1, 1)
    # points = np.concatenate((points_x, points_y, points_z), axis=1)
    # min_x, min_y, min_z = np.min(points, axis=0)
    # max_x, max_y, max_z = np.max(points, axis=0)
    # vertices = np.array(
    #     [[min_x, 0, min_z], [min_x, 0, max_z], [max_x, 0, max_z], [max_x, 0, min_z]],
    #     dtype=np.float32,
    # )
    # faces = np.array([[0, 1, 2], [0, 2, 3]])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def retrieve_furniture_mesh(
    detailed_obj_class,
    dec_objs_grained,
    fur_id,
    fur_cat,
    sdf_dir,
    denormalized_boxes,
    angles_pred,
    shapes_pred,
):
    class_name_grained = detailed_obj_class[int(dec_objs_grained[fur_id])]
    # print(class_name_grained)
    sdf_names = list(fur_cat[class_name_grained].keys())

    mesh_loaded = False
    attempts = 0
    max_attempts = len(sdf_names)

    # Attempt to load a mesh for the furniture, with a limit on the number of attempts
    while not mesh_loaded and attempts < max_attempts:
        sdf_name = random.choice(sdf_names)
        sdf_path = os.path.join(sdf_dir, sdf_name, "sdf.ply")
        if os.path.exists(sdf_path):
            mesh = trimesh.load(sdf_path)
            _, mesh = fit_shapes_to_box_3dfront(
                mesh,
                box_and_angle=torch.cat(
                    [denormalized_boxes[fur_id].float(), angles_pred[fur_id].float()]
                ),
                degrees=True,
            )
            shapes_pred[fur_id] = mesh
            mesh_loaded = True
            # Remove the attempted sdf_name to avoid repeating the same failed attempt
            sdf_names.remove(sdf_name)
            attempts += 1


def create_scene_meshes(
    dec_objs_grained,
    obj_to_pidx,
    denormalized_boxes,
    angles_pred,
    detailed_obj_class,
    fur_cat,
    sdf_dir,
    retrieve_sdf=True,
):
    # draw from furniture id
    num_obj = dec_objs_grained.size(0)
    shapes_pred = [0.0] * num_obj
    fur_idxs = [i for i in range(num_obj - 1) if i != obj_to_pidx[i]]
    room_idxs = [i for i in range(num_obj) if i not in fur_idxs]

    for fur_id in fur_idxs:
        if retrieve_sdf:
            retrieve_furniture_mesh(
                detailed_obj_class,
                dec_objs_grained,
                fur_id,
                fur_cat,
                sdf_dir,
                denormalized_boxes,
                angles_pred,
                shapes_pred,
            )
        else:
            box_and_angle = torch.cat(
                [denormalized_boxes[fur_id].float(), angles_pred[fur_id].float()]
            )
            box_points = params_to_8points_3dfront(box_and_angle, degrees=True)
            vertices, faces = box_vertices_faces(box_points)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            shapes_pred[fur_id] = mesh
    for room_id in room_idxs:
        mesh = make_room_mesh(denormalized_boxes[room_id], angles_pred[room_id])
        shapes_pred[room_id] = mesh
    return shapes_pred


def export_scene_meshes(shapes_pred, dec_objs, obj_idx2name, exp_path):

    # Create an empty scene
    scene = trimesh.Scene()

    # Add each mesh in your array to the scene
    for i, mesh in enumerate(shapes_pred):
        name = obj_idx2name[int(dec_objs[i])]
        scene.add_geometry(mesh, node_name=name)

    # Export the scene to an OBJ file
    scene.export(file_obj=exp_path, file_type="obj")


def render_plotly_sdf(
    box_points,
    box_and_angle=None,
    obj_idx2name=None,
    objs=None,
    shapes_pred=None,
    render_shapes=True,
    render_boxes=True,
    colors=None,
    save_as_image=True,
    filename="render",
    obj_to_pidx=None,
):
    fig = go.Figure()

    objs_id = objs.cpu().detach().numpy()
    unique_values = list(set(objs_id))

    if colors is None:
        # New color palette
        color_palette = {
            "hex": [
                "#8e7cc3ff",
                "#ea9999ff",
                "#93c47dff",
                "#9fc5e8ff",
                "#d55e00",
                "#cc79a7",
                "#c4458b",
                "#0072b2",
                "#f0e442",
                "#009e73",
            ]
        }
        value_to_color = {
            cls: color_palette["hex"][i % len(color_palette["hex"])]
            for i, cls in enumerate(unique_values)
        }
    else:
        with open(colors, "r") as file:
            obj_name_to_color = json.load(file)

    for i in range(len(box_points)):
        obj_id = int(objs[i])
        class_name = obj_idx2name[obj_id]
        pid = int(obj_to_pidx[i])
        points = box_points[i]

        vertices, faces = box_vertices_faces(points)
        if class_name in [
            "ceiling",
            "door",
            "doorframe",
        ]:
            continue

        if render_shapes:
            shape = shapes_pred[i]
            if i == pid:
                s_vertices, s_faces = shape.vertices, shape.faces

            else:
                if not box_and_angle:
                    print(
                        "fitting sdf shapes need box and angle"
                    )  # TODO: save the use of angle
                    raise ValueError
                box_points, denorm_shape = fit_shapes_to_box_3dfront(
                    shape, box_and_angle, degrees=True
                )
                s_vertices, s_faces = (
                    denorm_shape.vertices,
                    denorm_shape.faces,
                )  # assume shape is a trimesh obj
                # render mesh
                fig.add_trace(
                    go.Mesh3d(
                        x=s_vertices[:, 0],
                        y=s_vertices[:, 1],
                        z=s_vertices[:, 2],
                        i=s_faces[:, 0],
                        j=s_faces[:, 1],
                        k=s_faces[:, 2],
                        color=obj_name_to_color[class_name],
                        opacity=1,
                        name=f"{class_name}",
                        showlegend=True,
                    )
                )
                fig.update_traces(
                    flatshading=False,
                    lighting=dict(specular=0.8, ambient=0.6, diffuse=0.9),
                    selector=dict(type="mesh3d"),
                )
        camera = dict(up=dict(x=0, y=1, z=0), eye=dict(x=2, y=2, z=-2))
        fig.update_layout(scene_camera=camera)
        fig.update_layout(margin=dict(r=5, l=5, b=5, t=5), height=800, width=800)
        fig.update_layout(scene_aspectmode="data")
        fig.update_layout(
            scene=dict(
                xaxis_showspikes=False,
                yaxis_showspikes=False,
                zaxis_showspikes=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        )
        fig.update_scenes(camera_projection_type="orthographic")

        # Add bounding boxes
        if render_boxes:
            if class_name not in ["_scene_", "wall"]:
                # Compute the centroid of the box_points

                centroid = np.mean(points, axis=0)

                # Compute vectors from centroid to each point
                vectors = points - centroid

                # Sort points based on y-value, z-value, and x-value
                sorted_indices = np.lexsort(
                    (vectors[:, 0], vectors[:, 2], vectors[:, 1])
                )
                points = points[sorted_indices]
                edges = [
                    (0, 1),
                    (0, 2),
                    (0, 4),
                    (1, 3),
                    (1, 5),
                    (2, 3),
                    (2, 6),
                    (3, 7),
                    (4, 5),
                    (4, 6),
                    (5, 7),
                    (6, 7),
                ]

                for j, edge in enumerate(edges):
                    p1, p2 = edge
                    fig.add_trace(
                        go.Scatter3d(
                            x=[points[p1][0], points[p2][0]],
                            y=[points[p1][1], points[p2][1]],
                            z=[points[p1][2], points[p2][2]],
                            mode="lines",
                            legendgroup=class_name,
                            showlegend=j == 0,
                            name=class_name,
                            line=dict(
                                color=obj_name_to_color.get(class_name, "#b4d4ff")
                            ),
                        )
                    )
                fig.add_trace(
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=obj_name_to_color[class_name],
                        opacity=0.1,
                        name=f"{class_name}",
                        showlegend=True,
                    )
                )

    if save_as_image:
        scope = PlotlyScope()
        with open(filename, "wb") as f:
            f.write(scope.transform(fig, format="png"))
        # fig.show()
    else:
        fig.show()



