import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# =========================
# Load your precomputed dataset and trained model
# =========================
df = pd.read_csv("all_75_habitable_exoplanets_scores.csv")
df['pl_name_lower'] = df['pl_name'].str.lower()  # for case-insensitive matching

# === Re-train METHI Ensemble Model using your dataset ===
features = ['pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad']
X = df[features]
y = df['habitability_score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train component models
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                   random_state=42, subsample=0.9, colsample_bytree=0.9)
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                   solver='adam', max_iter=2000, random_state=42)

rf.fit(X_scaled, y)
xgb.fit(X_scaled, y)
mlp.fit(X_scaled, y)

# Ensemble model
methi_model = VotingRegressor([('rf', rf), ('xgb', xgb), ('mlp', mlp)])
methi_model.fit(X_scaled, y)

# =========================
# Streamlit App
# =========================
st.title("🌌 METHI Habitability Score Explorer")

planet_name_input = st.text_input("Enter exoplanet name:")

if planet_name_input:
    planet_name_input_lower = planet_name_input.lower()

    if planet_name_input_lower in df['pl_name_lower'].values:
        # Exact match
        planet_info = df[df['pl_name_lower'] == planet_name_input_lower]
        st.success(f"Found: {planet_info.iloc[0]['pl_name']}")
        st.dataframe(planet_info)
    else:
        # Fuzzy match
        all_planet_names = df['pl_name'].tolist()
        best_match, score, _ = process.extractOne(planet_name_input, all_planet_names)

        if score >= 80:
            st.warning(f"Did you mean **{best_match}**? Showing closest match.")
            planet_info = df[df['pl_name'] == best_match]
            st.dataframe(planet_info)
        else:
            st.error("Planet not found in the dataset. Pulling live data from NASA Exoplanet Archive...")
            url = (
                "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
                f"query=select+pl_name,pl_rade,pl_insol,st_teff,st_mass,st_rad+from+pscomppars+where+pl_name='{planet_name_input}'&format=csv"
            )
            try:
                new_data = pd.read_csv(url)
                if not new_data.empty:
                    st.success(f"Live data found for {planet_name_input}!")
                    # Compute METHI score
                    new_X = new_data[features]
                    new_X_scaled = scaler.transform(new_X)
                    predicted_score = methi_model.predict(new_X_scaled)
                    new_data['METHI_score'] = np.clip(predicted_score, 0, 1).round(2)
                    st.dataframe(new_data)
                else:
                    st.error("No data available for this planet, even in NASA archives.")
            except Exception as e:
                st.error(f"Error fetching live data: {e}")

# Leaderboard of Top 10
st.subheader("🏆 Top 10 Most Habitable Planets")
top10 = df.sort_values(by='habitability_score', ascending=False).head(10)
top10_display = top10[['pl_name', 'habitability_score']].copy()
top10_display['habitability_score'] = top10_display['habitability_score'].round(2)
st.table(top10_display)

# Full dataset download
st.download_button(
    label="Download Full Dataset",
    data=df.to_csv(index=False),
    file_name="all_75_habitable_exoplanets_scores.csv",
    mime="text/csv",
)
