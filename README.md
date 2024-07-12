# XAI_API


Welcome to the XAI_API repository! This project provides an Explainable AI (XAI) API using Python Flask, MongoDB, and React.js. The API offers transparency and interpretability for machine learning models through various techniques such as SHAP, LIME, and feature importance. The project also includes advanced features such as counterfactual explanations, with a frontend visualization to show changes needed in input features to alter predictions.

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
