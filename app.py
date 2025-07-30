import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rapidfuzz import process
import requests
import urllib.parse

# ============================
# Load models and datasets
# ============================

@st.cache_resource
def load_model():
    return joblib.load("methi_model.pkl")

@st.cache_data
def load_top75():
    df = pd.read_csv("all_75_habitable_exoplanets_scores.csv")
    df['pl_name_lower'] = df['pl_name'].str.lower()
    return df

@st.cache_data
def load_full_classification():
    df = pd.read_csv("stage1_predictions.csv")
    df['pl_name_lower'] = df['pl_name'].str.lower()
    return df

model = load_model()
df = load_top75()
full_df = load_full_classification()

top10 = df.sort_values(by="predicted_habitability_score", ascending=False).head(10).reset_index(drop=True)
planet_names = set(df['pl_name_lower'])
full_planet_names = set(full_df['pl_name_lower'])

# ============================
# NASA fetch fallback
# ============================

@st.cache_data
def fetch_nasa_data(planet_name):
    query = f"select+pl_name,pl_rade,pl_insol,st_teff,st_mass,st_rad+from+pscomppars+where+pl_name='{planet_name}'"
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=csv"
    try:
        df = pd.read_csv(url)
        if not df.empty:
            return df.iloc[0].to_dict()
    except:
        return None
    return None

# ============================
# Prediction helper
# ============================

def predict_score(planet_data):
    features = ['pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad']
    X = np.array([[planet_data[feat] for feat in features]])
    return float(np.clip(model.predict(X)[0], 0, 1))

# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="METHI Habitability Tool", layout="centered")
st.title("\U0001F30D METHI: Machine-Learned Exoplanetary Habitability Index")

# Leaderboard
st.subheader("Top 10 Most Habitable Exoplanets")
display_top10 = top10[['pl_name', 'predicted_habitability_score']].copy()
display_top10.columns = ["Exoplanet", "METHI Habitability Score"]
st.table(display_top10.style.format({"METHI Habitability Score": "{:.2f}"}))

# Search
st.subheader("Search for an Exoplanet")
planet_input = st.text_input("Enter planet name (case-insensitive):", key="planet_search")

if planet_input:
    search_name = planet_input.strip().lower()

    if search_name in planet_names:
        planet_row = df[df['pl_name_lower'] == search_name].iloc[0]
        st.success(f"Found {planet_row['pl_name']} in top 75!")
        show_df = planet_row[['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad', 'predicted_habitability_score']].to_frame().T
        show_df.columns = ["Exoplanet", "Radius (Earth)", "Insolation", "Stellar Teff", "Stellar Mass", "Stellar Radius", "METHI Habitability Score"]
        st.write(show_df.reset_index(drop=True).style.format({"METHI Habitability Score": "{:.2f}"}))

    else:
        best_match, score, _ = process.extractOne(search_name, df['pl_name_lower'])
        if score > 85:
            planet_row = df[df['pl_name_lower'] == best_match].iloc[0]
            st.info(f"Did you mean: {planet_row['pl_name']}?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, show this one"):
                    show_df = planet_row[['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad', 'predicted_habitability_score']].to_frame().T
                    show_df.columns = ["Exoplanet", "Radius (Earth)", "Insolation", "Stellar Teff", "Stellar Mass", "Stellar Radius", "METHI Habitability Score"]
                    st.write(show_df.reset_index(drop=True).style.format({"METHI Habitability Score": "{:.2f}"}))
            with col2:
                if st.button("No, let me type again"):
                    st.session_state.pop("planet_search", None)
                    st.rerun()

        elif search_name in full_planet_names:
            full_row = full_df[full_df['pl_name_lower'] == search_name].iloc[0]
            if full_row.get('habitable', 0) == 0:
                st.warning(f"{full_row['pl_name']} is classified as non-habitable in the METHI model.")
                show_df = full_row[['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad']].to_frame().T
                show_df.columns = ["Exoplanet", "Radius (Earth)", "Insolation", "Stellar Teff", "Stellar Mass", "Stellar Radius"]
                st.write(show_df.reset_index(drop=True))
            else:
                st.info(f"{full_row['pl_name']} is habitable but not in the top 75.")
                show_df = full_row[['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad']].to_frame().T
                show_df.columns = ["Exoplanet", "Radius (Earth)", "Insolation", "Stellar Teff", "Stellar Mass", "Stellar Radius"]
                st.write(show_df.reset_index(drop=True))

        else:
            st.warning(f"'{planet_input}' not found in dataset. Attempting NASA Archive fetch...")
            live_data = fetch_nasa_data(planet_input)
            if live_data:
                try:
                    score = predict_score(live_data)
                    st.success(f"Live METHI score for {planet_input}: {score:.2f}")
                    display_live = pd.DataFrame([live_data])
                    display_live['METHI Habitability Score'] = score
                    display_live = display_live[['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad', 'METHI Habitability Score']]
                    display_live.columns = ["Exoplanet", "Radius (Earth)", "Insolation", "Stellar Teff", "Stellar Mass", "Stellar Radius", "METHI Habitability Score"]
                    st.write(display_live.style.format({"METHI Habitability Score": "{:.2f}"}))
                except Exception as e:
                    st.error("Fetched data incomplete for METHI scoring.")
            else:
                st.error("No data found for this planet.")

# Dataset download
st.subheader("Download full habitability dataset")
download_df = df[['pl_name', 'predicted_habitability_score']].copy()
download_df.columns = ["Exoplanet", "METHI Habitability Score"]
csv = download_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name="METHI_scores.csv", mime="text/csv")
