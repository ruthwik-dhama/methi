import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from io import StringIO

# === Load Preprocessed Dataset ===
df = pd.read_csv("all_75_habitable_exoplanets_scores.csv")
hab_df = df.copy()

# === Standardize Features ===
features = ['pl_rade', 'pl_insol', 'st_teff', 'st_mass', 'st_rad']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(hab_df[features])

y = hab_df['habitability_score']
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === Train Models ===
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, subsample=0.9, colsample_bytree=0.9)
xgb.fit(X_train, y_train)

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)

ensemble = VotingRegressor([('rf', rf), ('xgb', xgb), ('mlp', mlp)])
ensemble.fit(X_train, y_train)
best_model = ensemble

# === Streamlit Interface ===
st.title("🌍 METHI: Exoplanet Habitability Estimator")
st.markdown("Enter an exoplanet name to see its habitability score.")

planet_name = st.text_input("🔭 Exoplanet Name:")

@st.cache_data
def fetch_exoplanet_data(planet_name):
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = (
        f"select pl_name,pl_rade,pl_insol,st_teff,st_mass,st_rad "
        f"from pscomppars where pl_name='{planet_name}'"
    )
    params = {"query": query, "format": "csv"}
    response = requests.get(base_url, params=params)

    if response.status_code == 200 and "pl_name" in response.text:
        df = pd.read_csv(StringIO(response.text))
        return df
    return None

if planet_name:
    result_row = hab_df[hab_df['pl_name'].str.lower() == planet_name.lower()]

    if result_row.empty:
        st.warning(f"'{planet_name}' not found in local dataset. Searching NASA...")
        live_data = fetch_exoplanet_data(planet_name)

        if live_data is not None and not live_data.empty:
            try:
                X_live = scaler.transform(live_data[features])
                predicted_score = best_model.predict(X_live)[0]
                predicted_score = np.clip(predicted_score, 0, 1)

                st.success(f"Live METHI score for {planet_name}: {predicted_score:.2f}")
                st.write(live_data)
            except:
                st.error("Could not compute METHI score due to missing or incompatible data.")
        else:
            st.error("Planet not found in NASA Exoplanet Archive.")
    else:
        st.success(f"{planet_name} METHI score (local data): {result_row['habitability_score'].values[0]:.2f}")
        st.dataframe(result_row)

# === Leaderboard ===
st.subheader("🏆 Top 10 Most Habitable Exoplanets")
top10 = hab_df.sort_values(by="habitability_score", ascending=False).head(10)
st.dataframe(top10[['pl_name', 'habitability_score']])
st.bar_chart(top10.set_index('pl_name')['habitability_score'])
