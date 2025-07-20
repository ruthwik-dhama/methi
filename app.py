import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process  # for fuzzy matching
import requests

# Load your precomputed dataset
df = pd.read_csv("all_75_habitable_exoplanets_scores.csv")
df['pl_name_lower'] = df['pl_name'].str.lower()  # for case-insensitive matching

st.title("METHI: ML Exoplanetary Terrestrial Habitability Index)")

# --- User Input ---
planet_name_input = st.text_input("Enter exoplanet name:")

# --- Search functionality ---
if planet_name_input:
    planet_name_input_lower = planet_name_input.lower()

    if planet_name_input_lower in df['pl_name_lower'].values:
        # Exact match
        planet_info = df[df['pl_name_lower'] == planet_name_input_lower]
        st.success(f"Found: {planet_info.iloc[0]['pl_name']}")
        st.dataframe(planet_info)
    else:
        # Fuzzy match for "Did you mean...?"
        all_planet_names = df['pl_name'].tolist()
        best_match, score, _ = process.extractOne(planet_name_input, all_planet_names)

        if score >= 80:
            st.warning(f"Did you mean **{best_match}**? Showing closest match.")
            planet_info = df[df['pl_name'] == best_match]
            st.dataframe(planet_info)
        else:
            st.error("Planet not found in the dataset. Pulling live data from NASA Exoplanet Archive...")

            # Fetch live data from NASA if not in local dataset
            url = (
                "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
                f"query=select+pl_name,pl_rade,pl_insol,st_teff,st_mass,st_rad+from+pscomppars+where+pl_name='{planet_name_input}'&format=csv"
            )
            try:
                new_data = pd.read_csv(url)
                if not new_data.empty:
                    st.success(f"Live data found for {planet_name_input}!")
                    st.dataframe(new_data)
                    st.info("Run METHI model on this planet to generate a habitability score.")
                else:
                    st.error("No data available for this planet, even in NASA archives.")
            except Exception as e:
                st.error(f"Error fetching live data: {e}")

# --- Leaderboard of Top 10 Habitable Planets ---
st.subheader("🏆 Top 10 Most Habitable Planets")
top10 = df.sort_values(by='habitability_score', ascending=False).head(10)
top10_display = top10[['pl_name', 'habitability_score']].copy()
top10_display['habitability_score'] = top10_display['habitability_score'].round(2)
st.table(top10_display)

# --- Full dataset download ---
st.download_button(
    label="Download Full Dataset",
    data=df.to_csv(index=False),
    file_name="all_75_habitable_exoplanets_scores.csv",
    mime="text/csv",
)
