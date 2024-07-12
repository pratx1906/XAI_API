def get_feature_importance(model):
    try:
        return model.feature_importances_.tolist()  # Convert to list for JSON serialization
    except AttributeError:
        raise Exception("Model does not have feature importance attribute.")
