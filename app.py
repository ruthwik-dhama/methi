import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained ensemble model (update path if needed)
model = joblib.load("methi_best_model.pkl")  # Replace with your actual model file
scaler = joblib.load("methi_scaler.pkl")     # StandardScaler used during training
cluster_model = joblib.load("methi_cluster_model.pkl")  # Clustering model from Stage 2

st.set_page_config(page_title="METHI Exoplanet Habitability Tool", layout="wide")

st.title("🌌 METHI: Machine-Learned Exoplanetary Habitability Index")
st.write("Input planetary and stellar data to get the METHI habitability score and classification.")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    pl_rade = st.number_input("Planet Radius (Earth radii)", value=1.0, format="%.3f")
    pl_bmasse = st.number_input("Planet Mass (Earth masses)", value=1.0, format="%.3f")
    pl_insol = st.number_input("Insolation Flux (Earth units)", value=1.0, format="%.3f")
    pl_orbeccen = st.number_input("Orbital Eccentricity", value=0.0, min_value=0.0, max_value=1.0, format="%.3f")
    pl_orbper = st.number_input("Orbital Period (days)", value=365.0, format="%.2f")

with col2:
    st_teff = st.number_input("Stellar Effective Temperature (K)", value=5700.0, format="%.1f")
    st_mass = st.number_input("Stellar Mass (Solar masses)", value=1.0, format="%.3f")
    st_rad = st.number_input("Stellar Radius (Solar radii)", value=1.0, format="%.3f")
    st_logg = st.number_input("Stellar Surface Gravity (log g)", value=4.4, format="%.2f")
    st_lum = st.number_input("Stellar Luminosity (Solar units)", value=1.0, format="%.3f")

if st.button("Predict Habitability"):
    # Prepare input
    features = np.array([[pl_rade, pl_insol, st_teff, st_mass, st_rad]])
    features_scaled = scaler.transform(features)

    # Predict habitability score
    methi_score = model.predict(features_scaled)[0]
    methi_score = np.clip(methi_score, 0, 1)

    # Cluster assignment
    cluster = cluster_model.predict(features_scaled)[0]

    # Display results
    st.subheader("METHI Prediction Results")
    st.metric("Habitability Score", f"{methi_score:.2f} / 1.0")
    st.write(f"**Cluster Assignment:** Cluster {cluster}")

    # Visualization: Radar plot
    feature_names = ['pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad']
    values = [pl_rade, pl_insol, st_teff, st_mass, st_rad]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    ax.plot(angles, values, color='teal', linewidth=2)
    ax.fill(angles, values, color='teal', alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_names)
    st.pyplot(fig)

    st.success("Prediction complete!")

st.info("METHI incorporates improved features beyond SEPHI 2.0, including better machine learning, clustering, and probabilistic habitability scoring.")
