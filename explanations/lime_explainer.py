import lime.lime_tabular
import numpy as np
import lime.lime_tabular


def get_lime_explanation(model, input_data, feature_names, class_names, training_data):
    """
    Computes LIME explanation for the given input data using the provided model.

    Args:
        model: The trained model for which LIME explanation is to be computed.
        input_data (list or numpy array): The input data for which LIME explanation needs to be computed.
        feature_names (list): The list of feature names.
        class_names (list): The list of class names.
        training_data (list or numpy array): The training data used for creating the LIME explainer.

    Returns:
        list: The LIME explanation for the input data.

    Raises:
        ValueError: If input data, feature names, class names, or training data are not in the expected format.
        Exception: For any other errors encountered during LIME explanation computation.
    """
    try:
        # Ensure training_data is a NumPy array
        training_data = np.array(training_data)
    except Exception as e:
        raise ValueError(f"Invalid training data format: {e}")

    try:
        # Initialize LIME Tabular Explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True
        )
        # Compute LIME explanation
        exp = explainer.explain_instance(np.array(input_data), model.predict_proba, num_features=len(feature_names))
        return exp.as_list()
    except Exception as e:
        raise Exception(f"Error computing LIME explanation: {e}")
