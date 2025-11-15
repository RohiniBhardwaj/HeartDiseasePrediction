import streamlit as st
import joblib
import numpy as np

# Load trained model + scaler
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease presence.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3, value=1)
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol Level", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.number_input("Resting ECG (0–2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0)
slope = st.number_input("Slope of Peak Exercise ST Segment (0–2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thalassemia (0–3)", min_value=0, max_value=3, value=2)

# Predict button
if st.button("Predict"):
    # Arrange features in correct order
    features = [
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]

    # Scale + predict
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    # Display result
    if prediction == 1:
        st.error("⚠ High chance of Heart Disease")
    else:
        st.success("✔ Low chance of Heart Disease")