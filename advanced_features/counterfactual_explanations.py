import random
import markovify
import numpy as np

# Sample text to train the Markov model (expanded sample sentences)
sample_text = """
To change the prediction, feature1 was modified from {original1} to {modified1}, 
feature2 from {original2} to {modified2}, feature3 from {original3} to {modified3}, 
and feature4 from {original4} to {modified4}.
Feature1 was increased from {original1} to {modified1}, and feature2 was decreased from {original2} to {modified2}.
The original value of feature3 was {original3}, which was changed to {modified3}.
Similarly, feature4's value was altered from {original4} to {modified4}.
"""

# Build the Markov model
text_model = markovify.Text(sample_text)

def generate_counterfactuals(model, input_data):
    if not isinstance(input_data, (list, np.ndarray)) or len(input_data) != 4:
        raise ValueError("Input data must be a list or numpy array of length 4.")

    original_prediction = model.predict([input_data])[0]

    counterfactual_data = {
        'feature1': input_data[0] + random.choice([-1, 1]),
        'feature2': input_data[1] + random.choice([-1, 1]),
        'feature3': input_data[2] + random.choice([-1, 1]),
        'feature4': input_data[3] + random.choice([-1, 1])
    }

    while model.predict([list(counterfactual_data.values())])[0] == original_prediction:
        counterfactual_data = {
            'feature1': input_data[0] + random.choice([-1, 1]),
            'feature2': input_data[1] + random.choice([-1, 1]),
            'feature3': input_data[2] + random.choice([-1, 1]),
            'feature4': input_data[3] + random.choice([-1, 1])
        }

    explanation_template = text_model.make_sentence()
    if explanation_template is None:
        explanation_template = (
            "To change the prediction, feature1 was modified from {original1} to {modified1}, "
            "feature2 from {original2} to {modified2}, feature3 from {original3} to {modified3}, "
            "and feature4 from {original4} to {modified4}."
        )

    counterfactual_explanation = explanation_template.format(
        original1=input_data[0], modified1=counterfactual_data['feature1'],
        original2=input_data[1], modified2=counterfactual_data['feature2'],
        original3=input_data[2], modified3=counterfactual_data['feature3'],
        original4=input_data[3], modified4=counterfactual_data['feature4']
    )

    return {
        'counterfactual_data': counterfactual_data,
        'explanation': counterfactual_explanation
    }
