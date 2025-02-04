import numpy as np
import pandas as pd  # Add this import for pandas
import pickle

# Load the trained model
loaded_model = pickle.load(open('D:/ML/trained_model_.sav', 'rb'))

# Define feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg','thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Input data (replace this with your actual input data)
input_data = [57, 1, 2, 150, 276, 0, 1, 112, 1, 1.2, 1, 0, 2]

# Create DataFrame
input_data_df = pd.DataFrame([input_data], columns=feature_names)

# Make prediction
prediction = loaded_model.predict(input_data_df)

# Output the prediction result
print(prediction)
if prediction[0] == 0:
    print('Congrats! You don’t have a heart disease.')
else:
    print('Sorry! The person has heart disease.')
