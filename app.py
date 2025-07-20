import pandas as pd
import numpy as np
import streamlit as st
import requests

# Load cached dataset if it exists
try:
    df = pd.read_csv("all_75_habitable_exoplanets_scores.csv")
except FileNotFoundError:
    df = pd.DataFrame()

st.set_page_config(page_title="METHI Habitability Scorer", layout="centered")
st.title("🌍 METHI: Exoplanet Habitability Scorer")

st.markdown("Enter an exoplanet name to retrieve its habitability score using METHI, or explore the top 10 habitable worlds.")

# === Leaderboard ===
if not df.empty:
    st.subheader("🏆 Top 10 Most Habitable Exoplanets")
    top10 = df.sort_values(by="predicted_habitability_score", ascending=False).head(10)
    top10_display = top10[["pl_name", "predicted_habitability_score"]].copy()
    top10_display["predicted_habitability_score"] = top10_display["predicted_habitability_score"].round(2)
    st.dataframe(top10_display, use_container_width=True)

# === User Input ===
st.subheader("🔍 Check Habitability for a Planet")

planet_name = st.text_input("Enter Exoplanet Name (e.g., TOI-700 d)").strip()

if planet_name:
    if planet_name in df['pl_name'].values:
        row = df[df['pl_name'] == planet_name].iloc[0]
        st.success(f"✅ {planet_name} found in local dataset!")
    else:
        # Fetch live data
        query = f"select+pl_name,pl_rade,pl_insol,st_teff,st_mass,st_rad+from+pscomppars+where+pl_name='{planet_name}'"
        url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=csv"
        try:
            live_df = pd.read_csv(url)
            if len(live_df) == 0:
                st.error("❌ Planet not found in NASA Exoplanet Archive.")
            else:
                row = live_df.iloc[0]
                # Calculate METHI proxy score
                radius, flux = row['pl_rade'], row['pl_insol']
                proxy_score = 1 - abs(radius - 1) / 2.5 - abs(flux - 1) / 2.0
                proxy_score = max(0, min(1, proxy_score))
                row["predicted_habitability_score"] = round(proxy_score, 2)
                st.success("✅ Planet found from NASA Exoplanet Archive!")
        except Exception as e:
            st.error("🚨 Error fetching planet data. Please check spelling or try again later.")
            st.stop()

    # Display planet details
    st.markdown("### 🌐 Planet Details")
    st.write({k: row[k] for k in ['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad'] if k in row})
    st.metric("🌱 METHI Habitability Score", round(row["predicted_habitability_score"], 2))
