# XAI_API
This API provides various explainable AI (XAI) functionalities, including prediction, SHAP explanations, LIME explanations, feature importance, and counterfactual explanations.

# Next Update 
Performance Optimization
Backend Database Optimization 
Integrated Gradients
DeepLIFT
Cache Layering

Endpoints
Predict Endpoint
URL: /predict

Method: POST

Description: Returns the prediction for the given input data.

Request:

Headers:
Content-Type: application/json
Body:
input: A list of numerical features required for prediction.
Response:

200 OK: Returns the predicted value.
400 Bad Request: Invalid input format.
500 Internal Server Error: Error in prediction.
SHAP Explanation Endpoint
URL: /explain/shap

Method: POST

Description: Returns SHAP values for the given input data.

Request:

Headers:
Content-Type: application/json
Body:
input: A list of numerical features for SHAP explanation.
Response:

200 OK: Returns the SHAP explanation.
400 Bad Request: Invalid input format.
500 Internal Server Error: Error in SHAP explanation.
LIME Explanation Endpoint
URL: /explain/lime

Method: POST

Description: Returns LIME explanation for the given input data.

Request:

Headers:
Content-Type: application/json
Body:
input: A list of numerical features for LIME explanation.
feature_names: A list of feature names.
class_names: A list of class names.
training_data: Training data used for the LIME explanation.
Response:

200 OK: Returns the LIME explanation.
400 Bad Request: Invalid input format.
500 Internal Server Error: Error in LIME explanation.
Feature Importance Endpoint
URL: /explain/feature_importance

Method: POST

Description: Returns the feature importance values of the loaded model.

Request:

Headers:
Content-Type: application/json
Body: None
Response:

200 OK: Returns the feature importance values.
500 Internal Server Error: Error in fetching feature importance.
Counterfactual Explanation Endpoint
URL: /explain/counterfactual

Method: POST

Description: Returns counterfactual explanations for the given input data.

Request:

Headers:
Content-Type: application/json
Body:
input: A list of numerical features for generating counterfactual explanations.
Response:

200 OK: Returns the counterfactual data and explanation.
500 Internal Server Error: Error in generating counterfactual explanation.
Error Handling
400 Bad Request: Returned when the request format is invalid.
500 Internal Server Error: Returned when there is an error processing the request on the server side.


##Running the API

Clone the repository:
git clone https://github.com/yourusername/XAI_API.git

Navigate to the project directory:
cd XAI_API

Create a virtual environment and activate it:
python -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`


Install the required packages:
pip install -r requirements.txt
Run the Flask application:


flask run
The API will be available at http://127.0.0.1:5000/.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Structure

```plaintext
XAI_API/
├── .idea/
├── advanced_features/
│   ├── counterfactual_explanations.py
├── explanations/
│   ├── feature_importance.py
│   ├── lime_explainer.py
│   ├── shap_explainer.py
├── models/
│   ├── model.pkl
│   ├── model.py
├── app.py
├── generate_model.py
├── Procfile
├── requirements.txt
├── runtime.txt
