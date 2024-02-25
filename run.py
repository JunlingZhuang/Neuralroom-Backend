import subprocess
import re
import json
from flask import Flask, jsonify
from flask_cors import CORS
from pathlib import Path


app = Flask(__name__)

CORS(app)

@app.route('/')
def test():
    return 'Hello World!'

@app.route('/happy') 
def happy():
    return 'Happy!'

@app.route('/generate')
def generate_model():
    print("generate_model")
    result = subprocess.run(['python', 'visualize_box_version.py'], capture_output=True, text=True)
    print(f'{result.stdout}')
    output = result.stdout
    match = re.search('<start>(.*?)<end>', output, re.DOTALL)
    print(match)
    if match:
        print('match found')
        model_file_path = Path(match.group(1))  # Get model file path
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
    app.run(debug=True)
