from graphviz import Digraph
import os
from helpers import viz_util
import json


# def visualize_scene_graph(graph, relationships, rel_filter_in = [], rel_filter_out = [], obj_ids = [], title ="", scan_id="",
# 													outfolder="./vis_graphs/"):
# 	g = Digraph(comment='Scene Graph' + title, format='png')

# 	for (i,obj) in enumerate(graph["objects"]):
# 		if (len(obj_ids) == 0) or (int(obj['id']) in obj_ids):
# 			if "node_mask" in graph.keys() and graph["node_mask"][i] == 0:
# 				g.node(str(obj['id']), obj["label"], fontname='helvetica', color=obj["ply_color"], fontcolor='red')
# 			else:
# 				g.node(str(obj['id']), obj["label"], fontname='helvetica', color=obj["ply_color"], style='filled')
# 	if "edge_mask" in graph.keys():
# 		edge_mask = graph["edge_mask"]
# 	else:
# 		edge_mask = None
# 	draw_edges(g, graph["relationships"], relationships, rel_filter_in, rel_filter_out, obj_ids, edge_mask)
# 	g.render(outfolder + scan_id)

from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import json


def visualize_scene_graph(
    rel_triples,
    objs,
    obj_idx2name,
    rel_idx2name,
    included_relations=[
        "left",
        "front",
        "standing on",
        "above",
        "inside",
    ],
    excluded_objs=["Room", "room"],
    obj_class_to_color=None,
):
    # Create a directed graph
    graph = Digraph(engine="dot")
    graph.attr(dpi="300")
    graph.attr(size="8,8")

    # Convert obj list to numpy for easy indexing

    if obj_class_to_color is None:
        # Generate a color palette based on object classes
        unique_obj_classes = np.unique(objs)
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
        obj_class_to_color = {
            cls: color_palette["hex"][i % len(color_palette["hex"])]
            for i, cls in enumerate(unique_obj_classes)
        }
    else:
        with open(obj_class_to_color, "r") as file:
            obj_name_to_color = json.load(file)
        # obj_name2idx = {v: k for k, v in obj_idx2name.items()}
        # obj_class_to_color = {
        #     obj_name2idx[key]: value
        #     for key, value in obj_name_to_color.items()
        #     if key in obj_name2idx
        # }
    # Iterate through relationship triples
    for rel in rel_triples:
        obj1_idx, rel_idx, obj2_idx = rel  # Convert tensor to list

        # Map index to class
        obj1_label = obj_idx2name[int(objs[obj1_idx])]
        obj2_label = obj_idx2name[int(objs[obj2_idx])]
        rel_label = rel_idx2name[int(rel_idx)]

        # Filter objects if excluded_objs is specified
        if excluded_objs and (
            obj1_label in excluded_objs or obj2_label in excluded_objs
        ):
            print("skip")
            continue

        # Filter relationships if included_relations is specified
        if rel_label not in included_relations:
            continue

        # Get names and colors from classes
        def format_object_name(name):
            return " ".join(word.capitalize() for word in name.split("_"))

        obj1_name = format_object_name(obj1_label)
        obj2_name = format_object_name(obj2_label)
        obj1_color = obj_name_to_color.get(obj1_label, "#b4d4ff")
        obj2_color = obj_name_to_color.get(obj2_label, "#b4d4ff")

        # Create node IDs combining the index and class for uniqueness
        node_id_1 = f"{obj1_idx}_{obj1_label}"
        node_id_2 = f"{obj2_idx}_{obj2_label}"

        # Add nodes

        graph.node(
            node_id_1,
            obj1_name,
            style="filled",
            fillcolor=obj1_color,
            fontname="Sans",
            fontsize="15",
            shape="circle",
            penwidth="3",
        )
        graph.node(
            node_id_2,
            obj2_name,
            style="filled",
            fillcolor=obj2_color,
            fontname="Sans",
            fontsize="15",
            shape="circle",
            penwidth="3",
        )

        # Add edge
        graph.edge(
            node_id_1,
            node_id_2,
            label=rel_label,
            style="solid",
            color="#524a48",
            arrowhead="empty",
            arrowsize="1",
            fontname="Sans",
            fontcolor="#26201e",
            penwidth="1",
            splines="curve",
        )

    # Render the graph
    graph.render(filename="scene_graph", cleanup=True, format="png")

    # Display the graph with adjusted size
    plt.figure(figsize=(20, 20))  # Set the figure size to 800x800 pixels
    img = mpimg.imread("scene_graph.png")
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    os.remove("scene_graph.png")

def draw_edges(
    g,
    graph_relationships,
    relationships,
    rel_filter_in,
    rel_filter_out,
    obj_ids,
    edge_mask=None,
):
    edges = {}
    if edge_mask is not None:
        joined_edge_mask = {}
    for i, rel in enumerate(graph_relationships):
        rel_text = relationships[rel[2]]
        if (
            len(rel_filter_in) == 0 or (rel_text.rstrip() in rel_filter_in)
        ) and not rel_text.rstrip() in rel_filter_out:
            if (len(obj_ids) == 0) or ((rel[1] in obj_ids) and (rel[0] in obj_ids)):
                index = str(rel[0]) + "_" + str(rel[1])
                if index not in edges:
                    edges[index] = []
                    if edge_mask is not None:
                        joined_edge_mask[index] = []
                edges[index].append(rel[3])
                if edge_mask is not None:
                    joined_edge_mask[index].append(edge_mask[i])

    for i, edge in enumerate(edges):
        edge_obj_sub = edge.split("_")
        rels = ", ".join(edges[edge])
        if edge_mask is not None and 0 in joined_edge_mask[edge]:
            g.edge(
                str(edge_obj_sub[0]),
                str(edge_obj_sub[1]),
                label=rels,
                color="red",
                style="dotted",
            )
        else:
            g.edge(str(edge_obj_sub[0]), str(edge_obj_sub[1]), label=rels, color="grey")


def run(
    use_sampled_graphs=True,
    scan_id="4d3d82b6-8cf4-2e04-830a-4303fa0e79c7",
    split=None,
    with_manipulation=False,
    data_path="./GT",
    outfolder="./vis_graphs/",
    graphfile="graphs_layout.yml",
):

    if use_sampled_graphs:
        # use this option to customize your own graphs in the yaml format
        palette_json = os.path.join(data_path, "color_palette.json")
        color_palette = json.load(open(palette_json, "r"))["hex"]
        graph_yaml = os.path.join(data_path, graphfile)
    else:
        # use this option to read scene graphs from the dataset
        relationships_json = os.path.join(
            data_path, "relationships_validation_clean.json"
        )  # "relationships_train.json")
        objects_json = os.path.join(data_path, "objects.json")

    relationships = viz_util.read_relationships(
        os.path.join(data_path, "relationships.txt")
    )

    if use_sampled_graphs:
        rel_label_to_id = {}
        for i, r in enumerate(relationships):
            rel_label_to_id[r] = i
        graph = viz_util.load_semantic_scene_graphs_custom(
            graph_yaml, color_palette, rel_label_to_id, with_manipuation=False
        )
        if with_manipulation:
            graph_mani = viz_util.load_semantic_scene_graphs_custom(
                graph_yaml, color_palette, rel_label_to_id, with_manipuation=True
            )
    else:
        graph = viz_util.load_semantic_scene_graphs(relationships_json, objects_json)

    if split != "":
        scan_id = scan_id + "_" + split

    filter_dict_in = []
    filter_dict_out = (
        []
    )  # ["left", "right", "behind", "front", "same as", "same symmetry as", "bigger than", "lower than", "higher than", "close by"]
    for scan_id in [scan_id]:
        visualize_scene_graph(
            graph[scan_id],
            relationships,
            filter_dict_in,
            filter_dict_out,
            [],
            "v1",
            scan_id=scan_id,
            outfolder=outfolder,
        )
        if with_manipulation and use_sampled_graphs:
            # manipulation only supported for custom graphs
            visualize_scene_graph(
                graph_mani[scan_id],
                relationships,
                filter_dict_in,
                filter_dict_out,
                [],
                "v1",
                scan_id=scan_id + "_mani",
                outfolder=outfolder,
            )

    idx = [o["id"] for o in graph[scan_id]["objects"]]
    color = [o["ply_color"] for o in graph[scan_id]["objects"]]
    # return used colors so that they can be used for 3D model visualization
    return dict(zip(idx, color))
