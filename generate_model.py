from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load sample data
iris = load_iris()
X, y = iris.data, iris.target

# Train a simple model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to a file
model_path = os.path.join('models', 'model.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to '{model_path}'")
