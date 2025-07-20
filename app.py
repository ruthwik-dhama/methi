import streamlit as st
import pandas as pd
import numpy as np
import requests
from fuzzywuzzy import process
from io import StringIO

st.set_page_config(page_title="🌍 METHI: Machine-Learned Exoplanetary Habitability Index")
st.title(":earth_africa: METHI: Machine-Learned Exoplanetary Habitability Index")

# === Load Precomputed METHI Scores ===
try:
    df = pd.read_csv("all_75_habitable_exoplanets_scores.csv")
    df['pl_name_lower'] = df['pl_name'].str.lower()
except Exception as e:
    st.error("Failed to load precomputed habitability scores. Please ensure the CSV is present.")
    st.stop()

# === Input Field ===
planet_input = st.text_input("Enter an exoplanet name:", key="planet_search")
planet_name = planet_input.strip().lower()

# === Fuzzy Matching for Input Errors ===
suggested_name = None
if planet_name and planet_name not in df['pl_name_lower'].values:
    matches = process.extractOne(planet_name, df['pl_name_lower'].values)
    if matches and matches[1] >= 80:
        suggested_name = matches[0]

# === Handle Suggestion Logic ===
if suggested_name:
    suggested_display = df.loc[df['pl_name_lower'] == suggested_name, 'pl_name'].values[0]
    st.markdown(f"Did you mean: **{suggested_display}**?")
    col1, col2 = st.columns(2)
    with col1:
        use_suggestion = st.button("Yes, use this name")
    with col2:
        reject_suggestion = st.button("No, let me type again")

    if use_suggestion:
        planet_name = suggested_name
        st.session_state['planet_search'] = suggested_display
    elif reject_suggestion:
        st.stop()  # Wait for user input again

# === If Planet Found ===
if planet_name in df['pl_name_lower'].values:
    row = df[df['pl_name_lower'] == planet_name].iloc[0]
    st.success(f"**Habitability Score for {row['pl_name']}**")
    st.metric("METHI Score (Predicted)", f"{row['predicted_habitability_score']:.2f}")
    
    st.write("### Key Features")
    st.write(pd.DataFrame({
        'Feature': ['Planet Radius (R_Earth)', 'Stellar Insolation (S_Earth)', 'Star Temperature (K)', 'Star Mass (M_Sun)', 'Star Radius (R_Sun)'],
        'Value': [row['pl_rade'], row['pl_insol'], row['st_teff'], row['st_mass'], row['st_rad']]
    }))

# === Leaderboard ===
st.write("---")
st.subheader(":trophy: Top 10 Most Habitable Exoplanets")
top10 = df.sort_values(by='predicted_habitability_score', ascending=False).head(10)
st.dataframe(top10[['pl_name', 'predicted_habitability_score']], use_container_width=True)

# === If not in dataset: Fetch from NASA Exoplanet Archive ===
if planet_name and planet_name not in df['pl_name_lower'].values and not suggested_name:
    st.info("Not found in local dataset. Checking NASA Exoplanet Archive...")
    try:
        query_url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_rade,pl_insol,st_teff,st_mass,st_rad+from+pscomppars+where+lower(pl_name)+like+'%25{planet_name}%25'&format=csv"
        response = requests.get(query_url)
        remote_df = pd.read_csv(StringIO(response.text))

        if remote_df.empty:
            st.warning("Could not find this exoplanet in NASA's Exoplanet Archive.")
        else:
            planet_data = remote_df.iloc[0]  # use the first match
            st.success(f"Live Data for {planet_data['pl_name']}")

            # Predict using the same habitability proxy formula
            r = planet_data['pl_rade']
            s = planet_data['pl_insol']
            score = 1 - abs(r - 1)/2.5 - abs(s - 1)/2.0
            score = np.clip(score, 0, 1)

            st.metric("Estimated METHI Score", f"{score:.2f}")
            st.write("### Key Features")
            st.write(pd.DataFrame({
                'Feature': ['Planet Radius (R_Earth)', 'Stellar Insolation (S_Earth)', 'Star Temperature (K)', 'Star Mass (M_Sun)', 'Star Radius (R_Sun)'],
                'Value': [planet_data['pl_rade'], planet_data['pl_insol'], planet_data['st_teff'], planet_data['st_mass'], planet_data['st_rad']]
            }))
    except Exception as e:
        st.error("Failed to fetch live data. Please check your internet connection or try again later.")
