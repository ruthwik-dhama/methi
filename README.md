# METHI: Machine-Learned Exoplanetary Habitability Index

METHI (Machine-Learned Exoplanetary Habitability Index) is a machine learning framework that identifies, classifies, clusters, and ranks confirmed exoplanets based on their potential habitability. The project leverages publicly available NASA exoplanet data and applies supervised and unsupervised learning techniques to generate an interpretable habitability score.

---

## Overview

The search for potentially habitable exoplanets has traditionally relied on simple threshold-based metrics. METHI extends this approach by combining machine learning algorithms with astrophysical constraints to identify planets that most closely resemble Earth-like conditions.

The framework consists of four stages:

1. Habitability Classification
2. Candidate Clustering
3. Habitability Scoring
4. Visualization and Interpretation

---

## Features

- Uses the latest NASA Exoplanet Archive data
- Binary classification of habitable vs. non-habitable planets
- Random Forest optimization using GridSearchCV
- K-Means and Gaussian Mixture clustering
- Continuous habitability scoring
- PCA visualization of candidate clusters
- Feature importance analysis
- Exportable ranked catalog of habitable exoplanets

---

## Dataset

Primary Dataset:

- NASA Exoplanet Archive
  - Confirmed Planet Table (`pscomppars.csv`)

Reference Dataset:

- Planetary Habitability Laboratory (PHL) Exoplanet Catalog

Approximately 5,900 confirmed exoplanets were analyzed.

---

## Input Features

The initial model uses the following astrophysical parameters:

| Feature | Description |
|----------|-------------|
| `pl_rade` | Planet Radius (Earth radii) |
| `pl_insol` | Stellar Insolation Flux |
| `st_teff` | Stellar Effective Temperature (K) |
| `st_mass` | Stellar Mass (Solar Masses) |
| `st_rad` | Stellar Radius (Solar Radii) |

---

# Project Structure

```
METHI/
│
├── data/
│   ├── pscomppars.csv
│   ├── stage1_predictions.csv
│   ├── habitable_candidates.csv
│   └── all_75_habitable_exoplanets_scores.csv
│
├── notebooks/
│   ├── Stage1_Classification.ipynb
│   ├── Stage2_Clustering.ipynb
│   ├── Stage3_Scoring.ipynb
│   └── Stage4_Visualization.ipynb
│
├── figures/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── pca_clusters.png
│   └── score_distribution.png
│
├── models/
│   ├── random_forest_classifier.pkl
│   └── random_forest_regressor.pkl
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Methodology

## Stage 1 — Habitability Classification

A binary classifier identifies planets that satisfy broad habitability constraints.

### Label Definition

A planet is considered potentially habitable if:

- Radius:
  ```
  0.8 ≤ Planet Radius ≤ 2.5 Earth Radii
  ```

- Insolation:
  ```
  0.3 ≤ Stellar Flux ≤ 2.0 Earth Flux
  ```

Random Forest Classification was selected because it:

- Handles nonlinear relationships
- Is robust to noisy astronomical data
- Provides feature importance
- Performs well on tabular datasets

GridSearchCV was used for hyperparameter optimization.

---

## Stage 2 — Candidate Clustering

Only planets predicted as habitable proceed to clustering.

Algorithms evaluated:

- K-Means
- Gaussian Mixture Models (GMM)
- DBSCAN

Results:

| Model | Silhouette Score |
|--------|-----------------:|
| K-Means (k=2) | 0.3851 |
| GMM (k=2) | 0.3364 |

K-Means provided the highest cluster separation.

---

## Stage 3 — Habitability Scoring

Each candidate receives a continuous Machine-Learned Exoplanetary Habitability Index (METHI).

The scoring model combines astrophysical parameters into a normalized target and trains a Random Forest Regressor.

Performance:

| Metric | Value |
|---------|-------|
| R² | **0.8629** |
| RMSE | **0.0801** |
| MAE | **0.0596** |

Best Hyperparameters:

```python
{
    "n_estimators": 50,
    "max_depth": 5
}
```

---

## Stage 4 — Visualization

The final stage visualizes the latent structure of habitable candidates using Principal Component Analysis (PCA).

Visualizations include:

- PCA projection
- Cluster assignments
- Feature importance
- Score distribution
- Ranked candidate plots

---

# Results

- Total confirmed planets analyzed: ~5,900
- Potentially habitable candidates identified: 75
- Classification performed using Random Forest
- Clustering validated using silhouette analysis
- Continuous habitability scores generated for every candidate

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/METHI.git
```

Move into the project directory:

```bash
cd METHI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

Run each stage sequentially:

```bash
python Stage1_Classification.py
```

```bash
python Stage2_Clustering.py
```

```bash
python Stage3_Scoring.py
```

```bash
python Stage4_Visualization.py
```

---

# Example Output

```
Planet Name                  METHI Score

Planet A                     0.97
Planet B                     0.95
Planet C                     0.94
...
```

The final output is a ranked catalog of exoplanets according to predicted habitability.

---

# Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy
- Jupyter Notebook

---

# Future Work

- Incorporate atmospheric composition and metallicity
- Integrate planetary density and orbital eccentricity
- Explore deep learning architectures
- Develop uncertainty-aware predictions
- Create a web-based visualization dashboard
- Expand feature engineering using stellar evolution models

---

# Citation

If you use this work in research, please cite:

```
Dhama, R.
METHI: Machine-Learned Exoplanetary Habitability Index.
2026.
```

---

# License

This project is released under the MIT License.

---

# Author

**Ruthwik Dhama**

Astrophysics • Machine Learning • Data Science

Research interests include:

- Exoplanet Habitability
- Stellar Spectroscopy
- Astroinformatics
- Machine Learning for Astronomy

---

## Acknowledgments

This project utilizes publicly available data from:

- NASA Exoplanet Archive
- Planetary Habitability Laboratory (PHL)

Special thanks to the scientific community for providing open astronomical datasets that make reproducible exoplanet research possible.
