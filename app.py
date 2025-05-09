# app.py
import streamlit as st
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Housing Price Predictor")
st.title("🏠 Housing Price Prediction")

# Input fields
area = st.number_input("Area (sq ft)", min_value=300, step=100)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)

# Predict
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ₹ {prediction:,.2f}")
