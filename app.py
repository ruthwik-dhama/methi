from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your trained model
with open("methi_ensemble.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler if saved separately (else create same preprocessing pipeline)
scaler = StandardScaler()

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>METHI Habitability Score Predictor</h1>
    <form action="/predict" method="post">
      Planet Radius (R_earth): <input name="pl_rade"><br>
      Insolation (Earth flux): <input name="pl_insol"><br>
      Star Temperature (K): <input name="st_teff"><br>
      Star Mass (M_sun): <input name="st_mass"><br>
      Star Radius (R_sun): <input name="st_rad"><br>
      <input type="submit">
    </form>
    """

@app.route("/predict", methods=["POST"])
def predict():
    # Extract input values
    features = [
        float(request.form["pl_rade"]),
        float(request.form["pl_insol"]),
        float(request.form["st_teff"]),
        float(request.form["st_mass"]),
        float(request.form["st_rad"])
    ]
    X_scaled = scaler.fit_transform([features])  # scale input

    # Predict habitability score
    score = model.predict(X_scaled)[0]
    return f"<h2>Predicted METHI Habitability Score: {score:.2f}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
