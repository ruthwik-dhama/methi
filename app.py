import streamlit as st
import pandas as pd
import numpy as np
import requests

# === Title ===
st.set_page_config(page_title="METHI Habitability Tool")
st.title("🪐 METHI Exoplanet Habitability Score Calculator")
st.markdown("Enter the name of an exoplanet to check its habitability score, or explore the top 10 most habitable exoplanets.")

# Load cached dataset if it exists
expected_columns = ['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad', 'habitability_score', 'predicted_habitability_score']
try:
    df = pd.read_csv("all_75_habitable_exoplanets_scores.csv")
except FileNotFoundError:
    df = pd.DataFrame(columns=expected_columns)

# User input section
planet_name = st.text_input("🔍 Enter exoplanet name (case-sensitive):")

# --- METHI Score Calculator ---
def methi_score(row):
    base = 1 - abs(row['pl_rade'] - 1) / 2.5 - abs(row['pl_insol'] - 1) / 2.0
    return round(np.clip(base, 0, 1), 2)

# --- Planet Lookup and Scoring ---
if planet_name:
    planet_row = df[df['pl_name'] == planet_name]

    if not planet_row.empty:
        st.success("✅ Found in cached dataset.")
        st.dataframe(planet_row.reset_index(drop=True))

    else:
        st.info("🌐 Fetching live data from NASA Exoplanet Archive...")
        query_url = (
            f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
            f"query=select+pl_name,pl_rade,pl_insol,st_teff,st_mass,st_rad+"
            f"from+pscomppars+where+pl_name='{planet_name}'&format=csv"
        )

        try:
            live_df = pd.read_csv(query_url)
            if live_df.empty:
                st.error("❌ Planet not found in NASA Archive.")
            else:
                row = live_df.iloc[0]
                score = methi_score(row)
                result = pd.DataFrame({
                    'pl_name': [row['pl_name']],
                    'pl_rade': [row['pl_rade']],
                    'pl_insol': [row['pl_insol']],
                    'st_teff': [row['st_teff']],
                    'st_mass': [row['st_mass']],
                    'st_rad': [row['st_rad']],
                    'habitability_score': [score],
                    'predicted_habitability_score': [score]  # Placeholder
                })
                st.success("✅ Planet found and scored.")
                st.dataframe(result)

        except Exception as e:
            st.error(f"❌ Error fetching planet data: {e}")

# === Leaderboard ===
if not df.empty:
    st.subheader("🏆 Top 10 Most Habitable Exoplanets (METHI)")
    top10 = df.sort_values(by='habitability_score', ascending=False).head(10)
    st.dataframe(top10.reset_index(drop=True))
