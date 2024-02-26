import subprocess
import re
import json
from flask import Flask, jsonify
from flask_cors import CORS
from pathlib import Path

from scripts.visualize_box_version import prepare_dataset_and_model,generate_queried_unit_mesh

app = Flask(__name__)

CORS(app)

#Global variable
dataset = None
model = None
args = None

#model and dataset initialized when webpage mounted
args,model,dataset,_,_ = prepare_dataset_and_model()
print('model initialized')

@app.route('/')
def test():
    return 'Hello World!'

@app.route('/happy') 
def happy():
    return 'Happy!'

@app.route('/generate')
def generate_model():
    print("generate_model")
    model_file_path = generate_queried_unit_mesh(queried_idx=0,args_location="./test/partition_emb_box_250/args.json",args=args,model=model,train_dataset=dataset)
    if model_file_path:
        try:
            with open(model_file_path, 'r') as model_file:
                model_data = model_file.read()  # Read model file content
                # set key value pair, frontent will use this key to get the model data
                return jsonify({"model_data": model_data}), 200
        except FileNotFoundError:
            return jsonify({"error": "Model file not found"}), 404
    else:
        return jsonify({"error": "Model file path not found in script output"}), 404

if __name__ == '__main__':
    print("Starting server...")
    app.run(debug=True,host = 'localhost')
