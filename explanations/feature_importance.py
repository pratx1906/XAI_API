def get_feature_importance(model):
    return model.feature_importances_.tolist()  # Convert to list for JSON serialization
