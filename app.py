import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rapidfuzz import process  # faster than fuzzywuzzy
import requests

# ============================
# Load pre-trained METHI model & data
# ============================

@st.cache_resource
def load_model():
    return joblib.load("methi_model.pkl")  # trained in Colab and saved once

@st.cache_data
def load_data():
    df = pd.read_csv("all_75_habitable_exoplanets_scores.csv")
    df['pl_name_lower'] = df['pl_name'].str.lower()
    return df

model = load_model()
df = load_data()

# Precompute leaderboard
top10 = df.sort_values(by="predicted_habitability_score", ascending=False).head(10)

# Set of names for fast lookup
planet_names = set(df['pl_name_lower'])

# ============================
# NASA Exoplanet Archive fetch (cached)
# ============================

@st.cache_data
def fetch_nasa_data(planet_name):
    query = f"select+pl_name,pl_rade,pl_insol,st_teff,st_mass,st_rad+from+pscomppars+where+pl_name='{planet_name}'"
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=csv"
    try:
        new_df = pd.read_csv(url)
        if not new_df.empty:
            return new_df.iloc[0].to_dict()
    except Exception:
        return None
    return None

# ============================
# Predict METHI score
# ============================

def predict_score(planet_data):
    features = ['pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad']
    X = np.array([[planet_data[feat] for feat in features]])
    pred = model.predict(X)[0]
    return np.clip(pred, 0, 1)

# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="METHI Habitability Tool", layout="centered")
st.title("METHI: Machine-Learned Exoplanetary Habitability Index")

# Leaderboard
st.subheader("Top 10 Most Habitable Exoplanets")
st.table(top10[['pl_name', 'predicted_habitability_score']].round(2))

# Search section
st.subheader("Search for an Exoplanet")
planet_input = st.text_input("Enter planet name (case-insensitive):", key="planet_search")

if planet_input:
    search_name = planet_input.strip().lower()

    if search_name in planet_names:
        planet_row = df[df['pl_name_lower'] == search_name].iloc[0]
        st.success(f"Found {planet_row['pl_name']} in dataset!")
        st.write(planet_row[['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad',
                             'predicted_habitability_score']])
    else:
        # Fuzzy match
        best_match, score, _ = process.extractOne(search_name, df['pl_name_lower'])
        if score > 85:
            planet_row = df[df['pl_name_lower'] == best_match].iloc[0]
            st.info(f"Did you mean: {planet_row['pl_name']}?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, show this one"):
                    st.write(planet_row[['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad',
                                         'predicted_habitability_score']])
            with col2:
                if st.button("No, let me type again"):
                    if "planet_search" in st.session_state:
                        st.session_state.pop("planet_search")  # safely clears input
                    st.rerun()  # triggers rerun and resets the input box
        else:
            st.warning(f"'{planet_input}' not found. Fetching live data from NASA...")
            live_data = fetch_nasa_data(planet_input)
            if live_data:
                score = predict_score(live_data)
                st.success(f"Live METHI score for {planet_input}: {score:.2f}")
                st.write(pd.DataFrame([live_data]))
            else:
                st.error("No data found in NASA Exoplanet Archive.")

# Download full dataset
st.subheader("Download full habitability dataset")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name="METHI_scores.csv", mime="text/csv")
