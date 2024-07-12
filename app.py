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
    """
    Home route to check if the API is running.
    """
    return "XAI API is running"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to get a prediction from the model.
    Expects a JSON body with an 'input' key containing a list of 4 numerical features.

    Returns:
        JSON response with the predicted value or an error message.
    """
    data = request.json
    if not data or 'input' not in data or not isinstance(data['input'], list):
        return jsonify({'error': 'Invalid input format. Expected a JSON object with an "input" key containing a list.'}), 400

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
    """
    Endpoint to get SHAP values for the input data.
    Expects a JSON body with an 'input' key containing a list of numerical features.

    Returns:
        JSON response with the SHAP explanation or an error message.
    """
    data = request.json
    if not data or 'input' not in data or not isinstance(data['input'], list):
        return jsonify({'error': 'Invalid input format. Expected a JSON object with an "input" key containing a list.'}), 400

    try:
        model = load_model()
        explanation = get_shap_values(model, data['input'])[0].tolist()  # Assuming a single-class explanation
        return jsonify({'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain/lime', methods=['POST'])
def lime_explain():
    """
    Endpoint to get LIME explanation for the input data.
    Expects a JSON body with 'input', 'feature_names', 'class_names', and 'training_data' keys.

    Returns:
        JSON response with the LIME explanation or an error message.
    """
    data = request.json
    required_keys = ['input', 'feature_names', 'class_names', 'training_data']
    if not all(key in data and isinstance(data[key], list) for key in required_keys):
        return jsonify({'error': 'Invalid input format. Expected a JSON object with required keys containing lists.'}), 400

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
    """
    Endpoint to get feature importance values of the loaded model.

    Returns:
        JSON response with the feature importance values or an error message.
    """
    try:
        model = load_model()
        importance = get_feature_importance(model)
        return jsonify({'importance': importance})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain/counterfactual', methods=['POST'])
def counterfactual():
    """
    Endpoint to get counterfactual explanations for the input data.
    Expects a JSON body with an 'input' key containing a list of numerical features.

    Returns:
        JSON response with the counterfactual data and explanation or an error message.
    """
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
