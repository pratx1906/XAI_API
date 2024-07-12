import shap
import numpy as np


def get_shap_values(model, input_data):
    # Convert input data to NumPy array
    input_data = np.array(input_data).reshape(1, -1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    return shap_values
