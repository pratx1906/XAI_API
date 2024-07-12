import pickle
import os


def load_model():
    model_path = os.path.join('models', 'model.pkl')
    print(f"Loading model from: {model_path}")  # Debug print statement
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")  # Debug print statement
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully")  # Debug print statement
    return model
