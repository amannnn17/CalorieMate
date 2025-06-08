import streamlit as st
import pickle
import numpy as np
from sklearn.utils.validation import check_is_fitted

# Load the model and scaler
model = pickle.load(open('finalized_model.pickle', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))

# Optional debug check
check_is_fitted(model)

# Page configuration
st.set_page_config(page_title="Calories Burned Predictor", page_icon="🔥", layout="centered")

# Custom heading
st.markdown("""
    <h2 style='text-align: center; color: #FF4B4B;'>🔥 Calories Burned Prediction App 🔥</h2>
    <p style='text-align: center;'>Enter your workout details to estimate the calories you've burned!</p>
    <hr>
""", unsafe_allow_html=True)

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('🎂 Age (years)', 10, 100, 25)
    height = st.number_input('📏 Height (cm)', 100, 250, 175)
    gender = st.selectbox('🧑 Gender', ['Male', 'Female'])

with col2:
    weight = st.number_input('⚖️ Weight (kg)', 30, 200, 70)
    duration = st.number_input('⏱️ Duration (minutes)', 1, 500, 30)
    heart_rate = st.number_input('❤️ Heart Rate (bpm)', 40, 200, 75)

# Additional input
body_temp = st.number_input('🌡️ Body Temperature (°C)', 35.0, 42.0, 37.0)

# Convert gender
gender_numeric = 1 if gender == 'Male' else 0

# Prepare input
input_data = np.array([[age, weight, height, gender_numeric, duration, heart_rate, body_temp]])
scaled_data = scaler.transform(input_data)

# Predict button
if st.button('🚀 Predict Calories Burned'):
    prediction = model.predict(scaled_data)[0]

    st.success(f'🔥 Estimated Calories Burned: **{prediction:.2f} kcal**')

    # Show input summary
    with st.expander("📊 View Your Input Summary"):
        st.markdown(f"""
        - **Age:** {age} years  
        - **Weight:** {weight} kg  
        - **Height:** {height} cm  
        - **Gender:** {'Male' if gender_numeric == 1 else 'Female'}  
        - **Duration:** {duration} minutes  
        - **Heart Rate:** {heart_rate} bpm  
        - **Body Temp:** {body_temp} °C
        """)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
