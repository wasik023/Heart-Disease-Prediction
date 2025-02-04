import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
try:
    with open('trained_model_.sav', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def predict_heart_disease(input_data):
    # Convert input data to DataFrame
    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    input_data_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Make prediction
    prediction = loaded_model.predict(input_data_df)

    # Return prediction result
    return "Congrats! You don’t have heart disease. ✅" if prediction[0] == 0 else "Sorry! The person has heart disease. ❌"

def main():
    st.title('💓 Heart Disease Prediction Web App')
    st.write("Enter the following details to predict the likelihood of heart disease:")
    
    # Input fields for features
    age = st.number_input('Age in years', min_value=1, max_value=120, value=30)
    sex = st.radio('Sex', options=[(1, 'Male'), (0, 'Female')], format_func=lambda x: x[1])[0]
    
    cp = st.selectbox('Chest Pain Type', [
        (0, 'Typical angina'),
        (1, 'Atypical angina'),
        (2, 'Non-anginal pain'),
        (3, 'Asymptomatic')
    ], format_func=lambda x: x[1])[0]
    
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=200, value=120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.radio('Fasting Blood Sugar > 120 mg/dl?', [(1, 'Yes'), (0, 'No')], format_func=lambda x: x[1])[0]
    
    restecg = st.selectbox('Resting ECG Results', [
        (0, 'Normal'),
        (1, 'ST-T wave abnormality'),
        (2, 'Probable/definite LVH')
    ], format_func=lambda x: x[1])[0]
    
    thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.radio('Exercise-Induced Angina?', [(1, 'Yes'), (0, 'No')], format_func=lambda x: x[1])[0]
    oldpeak = st.slider('ST Depression (induced by exercise)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    slope = st.selectbox('Slope of ST Segment', [
        (0, 'Upsloping'),
        (1, 'Flat'),
        (2, 'Downsloping')
    ], format_func=lambda x: x[1])[0]
    
    ca = st.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=0)
    
    thal = st.selectbox('Thalassemia', [
        (1, 'Normal'),
        (2, 'Fixed defect'),
        (3, 'Reversible defect')
    ], format_func=lambda x: x[1])[0]

    # Prepare input data
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Predict button
    if st.button('🔍 Predict'):
        result = predict_heart_disease(input_data)
        st.success(result)

if __name__ == "__main__":
    main()
