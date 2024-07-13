from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import os
from models.model import load_model
from explanations.shap_explainer import get_shap_values
from explanations.lime_explainer import get_lime_explanation
from explanations.feature_importance import get_feature_importance
from advanced_features.counterfactual_explanations import generate_counterfactuals

app = Flask(__name__)
CORS(app)
client = MongoClient('mongodb://localhost:27017/')
db = client.xai_database

app.config['UPLOAD_FOLDER'] = 'models/'
ALLOWED_EXTENSIONS = {'pkl'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'model.pkl'))
        return jsonify({'message': 'Model uploaded successfully'}), 200
    return jsonify({'error': 'File not allowed'}), 400


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'input' not in data or not isinstance(data['input'], list):
        return jsonify(
            {'error': 'Invalid input format. Expected a JSON object with an "input" key containing a list.'}), 400
    if len(data['input']) != 4:
        return jsonify({'error': 'Invalid input format. Expected a list of 4 features.'}), 400

    try:
        model = load_model()
        prediction = model.predict([data['input']])[0].item()  # Convert to native Python type
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain/shap', methods=['POST'])
def shap_explain():
    data = request.json
    if not data or 'input' not in data or not isinstance(data['input'], list):
        return jsonify(
            {'error': 'Invalid input format. Expected a JSON object with an "input" key containing a list.'}), 400

    try:
        model = load_model()
        explanation = get_shap_values(model, data['input'])[0].tolist()  # Assuming a single-class explanation
        return jsonify({'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain/lime', methods=['POST'])
def lime_explain():
    data = request.json
    required_keys = ['input', 'feature_names', 'class_names', 'training_data']
    if not all(key in data and isinstance(data[key], list) for key in required_keys):
        return jsonify(
            {'error': 'Invalid input format. Expected a JSON object with required keys containing lists.'}), 400

    try:
        model = load_model()
        explanation = get_lime_explanation(
            model,
            data['input'],
            data['feature_names'],
            data['class_names'],
            data['training_data']
        )
        return jsonify({'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain/feature_importance', methods=['POST'])
def feature_importance():
    try:
        model = load_model()
        importance = get_feature_importance(model)
        return jsonify({'importance': importance})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain/counterfactual', methods=['POST'])
def counterfactual():
    data = request.json
    if not data or 'input' not in data or not isinstance(data['input'], list):
        return jsonify({'error': 'Invalid input format. Expected a JSON object with an "input" key containing a list.'}), 400

    try:
        model = load_model()
        counterfactuals = generate_counterfactuals(model, data['input'])
        return jsonify(counterfactuals)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
