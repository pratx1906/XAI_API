import pickle
import os

def load_model():
    """
    Loads and returns the trained model from a pickle file.

    The model is expected to be stored in 'models/model.pkl'. If the file does not
    exist, an error message is printed.

    Returns:
        model: The trained model loaded from the pickle file.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: For any other errors encountered during loading.
    """
    model_path = os.path.join('models', 'model.pkl')
    print(f"Loading model from: {model_path}")  # Debug print statement

    if not os.path.exists(model_path):
        error_message = f"Model file not found: {model_path}"
        print(error_message)  # Debug print statement
        raise FileNotFoundError(error_message)

    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully")  # Debug print statement
        return model
    except Exception as e:
        error_message = f"An error occurred while loading the model: {e}"
        print(error_message)  # Debug print statement
        raise e
