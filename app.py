import json
from flask import jsonify, request
from training.pipeline import training_pipeline, analysis_pipeline
import config
from flask_app import app

path = app.config.get('MODELS_FOLDER')

@app.route('/status', methods=['GET'])
def test():
    if request.method == 'GET':
        return 'App is running'

@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        r_data = request.get_json()
        data = json.loads(json.dumps(r_data))
        files_dict = data['files']
        labels = data['labels']
        config_file = config.read_config_file(data['config'])
        res = training_pipeline(files_dict, labels, path, config_file)

        return jsonify(res), 200

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        files_dict = data['files']
        model_name = data['modelFile']
        
        res = analysis_pipeline(files_dict, path, model_name)

        return jsonify(res), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0")
