import streamlit as st
import pandas as pd
import numpy as np
import requests
from fuzzywuzzy import process  # fuzzy string matching

st.set_page_config(page_title="METHI Habitability Scorer", layout="centered")

st.title("METHI: ML Exoplanetary Terrestrial Habitability Index")

# Load dataset
try:
    df = pd.read_csv("all_75_habitable_exoplanets_scores.csv")
except FileNotFoundError:
    df = pd.DataFrame()
    st.warning("⚠️ Local dataset not found. Only live lookup will be used.")

# Normalize planet names for matching
if not df.empty and "pl_name" in df.columns:
    df["pl_name_clean"] = df["pl_name"].str.strip().str.lower()

# Leaderboard
if not df.empty:
    st.subheader("🏆 Top 10 Most Habitable Exoplanets")
    top10 = df.sort_values("predicted_habitability_score", ascending=False).head(10)
    st.table(top10[["pl_name", "predicted_habitability_score"]])

# Input box
st.subheader("🔍 Search for a Planet")
planet_name = st.text_input("Enter Exoplanet Name").strip()

# Habitability proxy
def habitability_proxy(pl_rade, pl_insol):
    score = 1 - abs(pl_rade - 1)/2.5 - abs(pl_insol - 1)/2.0
    return round(np.clip(score, 0, 1), 2)

# Fetch live data
def fetch_from_nasa(name):
    query = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        f"query=select+pl_name,pl_rade,pl_insol,st_teff,st_mass,st_rad+"
        f"from+pscomppars+where+lower(pl_name)='{name.lower()}'&format=csv"
    )
    try:
        return pd.read_csv(query)
    except Exception:
        return pd.DataFrame()

if planet_name:
    cleaned = planet_name.lower()
    match = pd.DataFrame()

    if not df.empty and "pl_name_clean" in df.columns:
        # Exact match first
        match = df[df["pl_name_clean"] == cleaned]

        # Fuzzy match if no exact match
        if match.empty:
            best_match = process.extractOne(cleaned, df["pl_name_clean"], score_cutoff=80)
            if best_match:
                match = df[df["pl_name_clean"] == best_match[0]]

    if not match.empty:
        row = match.iloc[0]
        st.success(f"Found in dataset: {row['pl_name']}")
        st.write("Habitability Score:", row["predicted_habitability_score"])
        st.dataframe(row[["pl_rade", "pl_insol", "st_teff", "st_mass", "st_rad"]].T.rename(columns={row.name: "Value"}))
    else:
        st.info("Not in dataset. Fetching from NASA...")
        live = fetch_from_nasa(planet_name)
        if not live.empty and pd.notnull(live.iloc[0]["pl_rade"]) and pd.notnull(live.iloc[0]["pl_insol"]):
            row = live.iloc[0]
            score = habitability_proxy(row["pl_rade"], row["pl_insol"])
            st.success(f"Found: {row['pl_name']}")
            st.write("Estimated Habitability Score (proxy):", score)
            st.dataframe(row[["pl_rade", "pl_insol", "st_teff", "st_mass", "st_rad"]].T.rename(columns={row.name: "Value"}))
        else:
            st.error("No sufficient data found.")
