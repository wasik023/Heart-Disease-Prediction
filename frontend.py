import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('D:/ML/trained_model_.sav', 'rb'))

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
    if prediction[0] == 0:
        return 'Congrats! You don’t have heart disease.'
    else:
        return 'Sorry! The person has heart disease.'

def main():
    st.title('Heart Disease Prediction Web App')
    st.write("Enter the following details to predict the likelihood of heart disease:")
    st.markdown("""
    - **Sex:** `1` for Male, `0` for Female  
    - **Chest pain type (cp):**
        - `0`: Typical angina  
        - `1`: Atypical angina  
        - `2`: Non-anginal pain  
        - `3`: Asymptomatic  
    - **Fasting blood sugar (fbs):** `1` if > 120 mg/dl, `0` otherwise  
    - **Resting ECG (restecg):**
        - `0`: Normal  
        - `1`: ST-T wave abnormality  
        - `2`: Probable/definite left ventricular hypertrophy (LVH)  
    - **Exercise-induced angina (exang):** `1` for Yes, `0` for No  
    - **Slope of ST segment (slope):**
        - `0`: Upsloping  
        - `1`: Flat  
        - `2`: Downsloping  
    - **Thalassemia (thal):**
        - `1`: Normal  
        - `2`: Fixed defect  
        - `3`: Reversible defect  
    """)

    # Input fields for features
    age = st.number_input('Age in years', min_value=1, max_value=120, value=30)
    sex = st.selectbox('Sex (1 for Male, 0 for Female)', options=[(1, 'Male'), (0, 'Female')], format_func=lambda x: x[1])[0]
    cp = st.selectbox('Chest pain type', options=[
        (0, 'Typical angina'),
        (1, 'Atypical angina'),
        (2, 'Non-anginal pain'),
        (3, 'Asymptomatic')
    ], format_func=lambda x: x[1])[0]
    trestbps = st.number_input('Resting blood pressure (mm Hg)', min_value=50, max_value=200, value=120)
    chol = st.number_input('Serum cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting blood sugar > 120 mg/dl (1 for Yes, 0 for No)', options=[(1, 'Yes'), (0, 'No')], format_func=lambda x: x[1])[0]
    restecg = st.selectbox('Resting ECG results', options=[
        (0, 'Normal'),
        (1, 'ST-T wave abnormality'),
        (2, 'Probable/definite LVH')
    ], format_func=lambda x: x[1])[0]
    thalach = st.number_input('Maximum heart rate achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise-induced angina (1 for Yes, 0 for No)', options=[(1, 'Yes'), (0, 'No')], format_func=lambda x: x[1])[0]
    oldpeak = st.number_input('ST depression induced by exercise relative to rest', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope of peak exercise ST segment', options=[
        (0, 'Upsloping'),
        (1, 'Flat'),
        (2, 'Downsloping')
    ], format_func=lambda x: x[1])[0]
    ca = st.number_input('Number of major vessels colored by fluoroscopy (0-3)', min_value=0, max_value=3, value=0)
    thal = st.selectbox('Thalassemia', options=[
        (1, 'Normal'),
        (2, 'Fixed defect'),
        (3, 'Reversible defect')
    ], format_func=lambda x: x[1])[0]

    # Create the input data array
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Predict and display the result
    if st.button('Predict'):
        result = predict_heart_disease(input_data)
        st.success(result)

if __name__ == "__main__":
    main()
