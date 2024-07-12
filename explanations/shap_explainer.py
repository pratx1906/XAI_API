import shap
import numpy as np
import shap


def get_shap_values(model, input_data):
    """
    Computes SHAP values for the given input data using the provided model.

    Args:
        model: The trained model for which SHAP values are to be computed.
        input_data (list or numpy array): The input data for which SHAP values need to be computed.

    Returns:
        list: The SHAP values for the input data.

    Raises:
        ValueError: If input_data is not in the expected format.
        Exception: For any other errors encountered during SHAP value computation.
    """
    try:
        # Convert input data to NumPy array and reshape
        input_data = np.array(input_data).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Invalid input data format: {e}")

    try:
        # Initialize SHAP TreeExplainer and compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        return shap_values
    except Exception as e:
        raise Exception(f"Error computing SHAP values: {e}")
