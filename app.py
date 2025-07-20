import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model (replace with your actual model path)
model = joblib.load("methi_model.pkl")

st.title("METHI Exoplanet Habitability Predictor")

# Input fields for users
pl_rade = st.number_input("Planet Radius (Earth radii)", value=1.0)
pl_insol = st.number_input("Insolation Flux (Earth units)", value=1.0)
st_teff = st.number_input("Stellar Effective Temperature (K)", value=5700.0)
st_mass = st.number_input("Stellar Mass (Solar masses)", value=1.0)
st_rad = st.number_input("Stellar Radius (Solar radii)", value=1.0)

if st.button("Predict Habitability"):
    features = np.array([[pl_rade, pl_insol, st_teff, st_mass, st_rad]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Habitability Score: {prediction:.2f}")
