import subprocess
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

from scripts.visualize_box_version import (
    prepare_dataset_and_model,
    generate_queried_unit_mesh,
)

app = Flask(__name__)

CORS(app)

# Global variable
dataset = None
model = None
args = None

# model and dataset initialized when webpage mounted
args, model, dataset, _, _ = prepare_dataset_and_model(
    args_location="./test/partitionv2_simedge2_unit1_woCLIP_1500/args.json",
    ckpt_epoch=400,
)
print("model initialized")


@app.route("/")
def test():
    return "Hello World!"


@app.route("/happy")
def happy():
    return "Happy!"


@app.route('/generate',methods=['POST'])
def generate_model():
    print("generate_model")
    data = request.get_json()
    length = data.get('length')
    height = data.get('height')
    width = data.get('width')
    # set length, height, width to float if not None
    length = float(length) if length is not None else 0.0
    height = float(height) if height is not None else 0.0
    width = float(width) if width is not None else 0.0

    print(length, height, width)

    # get graph data
    objs = data.get('nodes')
    triples = data.get('edges')

    print(objs,triples)
    model_file_path = generate_queried_unit_mesh(
        input_objs=objs,
        input_triples=triples,
        unit_box = [length,height,width],
        args=args,
        model=model,
        train_dataset=dataset,
    )
    print(model_file_path)


    if model_file_path:
        try:
            with open(model_file_path, "r") as model_file:
                model_data = model_file.read()  # Read model file content
                # set key value pair, frontent will use this key to get the model data
                return jsonify({"model_data": model_data}), 200
        except FileNotFoundError:
            return jsonify({"error": "Model file not found"}), 404
    else:
        return jsonify({"error": "Model file path not found in script output"}), 404


if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True, host="localhost")
