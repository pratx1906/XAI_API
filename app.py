from flask import Flask, request, jsonify
from pymongo import MongoClient
from models.model import load_model
from explanations.shap_explainer import get_shap_values
from explanations.lime_explainer import get_lime_explanation
from explanations.feature_importance import get_feature_importance
from advanced_features.counterfactual_explanations import generate_counterfactuals

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client.xai_database


@app.route('/')
def index():
    return "XAI API is running"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'input' not in data or not isinstance(data['input'], list):
            return jsonify(
                {'error': 'Invalid input format. Expected a JSON object with an "input" key containing a list.'}), 400
        if len(data['input']) != 4:
            return jsonify({'error': 'Invalid input format. Expected a list of 4 features.'}), 400
        model = load_model()
        prediction = model.predict([data['input']])
        prediction = prediction[0].item()  # Convert to native Python type
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain/shap', methods=['POST'])
def shap_explain():
    try:
        data = request.json
        print(f"Received data for SHAP explanation: {data}")
        if 'input' not in data or not isinstance(data['input'], list):
            return jsonify({'error': 'Invalid input format. Expected a JSON object with an "input" key containing a '
                                     'list.'}), 400
        model = load_model()
        explanation = get_shap_values(model, data['input'])
        # Convert explanation to a serializable format if needed
        explanation = explanation[0].tolist()  # Assuming a single-class explanation
        return jsonify({'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain/lime', methods=['POST'])
def lime_explain():
    try:
        data = request.json
        if 'input' not in data or not isinstance(data['input'], list):
            return jsonify({'error': 'Invalid input format. Expected a JSON object with an "input" key containing a '
                                     'list.'}), 400
        if 'feature_names' not in data or not isinstance(data['feature_names'], list):
            return jsonify({'error': 'Invalid input format. Expected a JSON object with a "feature_names" key '
                                     'containing a list.'}), 400
        if 'class_names' not in data or not isinstance(data['class_names'], list):
            return jsonify({'error': 'Invalid input format. Expected a JSON object with a "class_names" key '
                                     'containing a list.'}), 400
        if 'training_data' not in data or not isinstance(data['training_data'], list):
            return jsonify({'error': 'Invalid input format. Expected a JSON object with a "training_data" key '
                                     'containing a list.'}), 400

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
    try:
        data = request.json
        model = load_model()
        counterfactuals = generate_counterfactuals(model, data['input'])
        return jsonify(counterfactuals)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
