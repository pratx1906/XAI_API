import lime.lime_tabular
import numpy as np


def get_lime_explanation(model, input_data, feature_names, class_names, training_data):
    # Ensure training_data is a NumPy array
    training_data = np.array(training_data)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True
    )
    exp = explainer.explain_instance(np.array(input_data), model.predict_proba, num_features=len(feature_names))
    return exp.as_list()
