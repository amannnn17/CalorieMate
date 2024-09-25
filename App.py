import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open('finalized_model.pickle', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))

# Streamlit app UI
st.title('Calories Burned Prediction App')

# Get user input for features
age = st.number_input('Enter your age (years)', min_value=10, max_value=100, value=25)
weight = st.number_input('Enter your weight (kg)', min_value=30, max_value=200, value=70)
height = st.number_input('Enter your height (cm)', min_value=100, max_value=250, value=175)
gender = st.selectbox('Select your gender', ['Male', 'Female'])
exercise_duration = st.number_input('Enter exercise duration (minutes)', min_value=1, max_value=500, value=30)

# New fields added
heart_rate = st.number_input('Enter your heart rate (bpm)', min_value=40, max_value=200, value=75)
steps_count = st.number_input('Enter your steps count', min_value=0, max_value=30000, value=5000)

# Convert gender to numeric value
gender_numeric = 1 if gender == 'Male' else 0

# Collect input features into a NumPy array (7 features)
input_data = np.array([[age, weight, height, gender_numeric, exercise_duration, heart_rate, steps_count]])

# Scale the input data using the loaded scaler
scaled_data = scaler.transform(input_data)

# When the user clicks "Predict"
if st.button('Predict'):
    # Make the prediction using the loaded model
    prediction = model.predict(scaled_data)
    
    # Display the prediction result
    st.success(f'Predicted Calories Burned: {prediction[0]:.2f} kcal')
