# METHI
### Machine-Learned Exoplanetary Habitability Index

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)](https://xgboost.ai)
[![IEEE](https://img.shields.io/badge/IEEE-CONMEDIA%202025-blue.svg)](https://ieeexplore.ieee.org/document/11290119)

---

## Publication

**METHI: An Ensemble-based Machine Learned Exoplanetary Habitability Index** was peer-reviewed and published in the **2025 International Conference on New Media Studies (CONMEDIA 2025)** and is available through the IEEE Xplore Digital Library.

**Authors**

- Ruthwik Dhama
- Praveen Pratap Singh
- Vishal Kumar

**IEEE Xplore:** https://ieeexplore.ieee.org/document/11290119

---

## Overview

**METHI (Machine-Learned Exoplanetary Habitability Index)** is a published machine learning framework designed to identify, classify, cluster, and rank potentially habitable exoplanets using observational data from the **NASA Exoplanet Archive**.

Traditional habitability indices such as the Earth Similarity Index (ESI), Planetary Habitability Index (PHI), and Statistical-likelihood Exoplanetary Habitability Index (SEPHI) rely primarily on deterministic equations and manually engineered heuristics. METHI extends these approaches by learning complex nonlinear relationships directly from planetary and stellar parameters using modern machine learning techniques.

The framework combines supervised classification, ensemble learning, unsupervised clustering, and continuous habitability scoring to produce an interpretable Machine-Learned Exoplanetary Habitability Index. An accompanying Streamlit application enables researchers and students to analyze exoplanets in real time through an interactive interface.

---

## Key Features

- Ensemble machine learning framework for exoplanet habitability prediction
- Random Forest, XGBoost, and Multi-Layer Perceptron (MLP) models
- Ensemble voting for improved predictive performance
- K-Means clustering of habitable candidates
- Continuous habitability scoring and ranking
- Interactive Streamlit web application
- Automated preprocessing and prediction pipeline
- Exportable prediction results
- Published IEEE research methodology

---

## Machine Learning Pipeline

```text
NASA Exoplanet Archive
            │
            ▼
     Data Preprocessing
            │
            ▼
    Feature Engineering
            │
            ▼
 ┌────────────────────────┐
 │ Random Forest Model    │
 └────────────────────────┘
            │
 ┌────────────────────────┐
 │ XGBoost Model          │
 └────────────────────────┘
            │
 ┌────────────────────────┐
 │ Neural Network (MLP)   │
 └────────────────────────┘
            │
            ▼
     Ensemble Learning
            │
            ▼
 Habitability Classification
            │
            ▼
   K-Means Clustering
            │
            ▼
 Continuous Habitability Score
            │
            ▼
 Interactive Streamlit Dashboard
```

---

## Repository Structure

```text
METHI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── Models
│   ├── methi_model.pkl
│   ├── methi_best_model.pkl
│   ├── methi_rf.pkl
│   ├── methi_xgb.pkl
│   ├── methi_mlp.pkl
│   ├── methi_ensemble.pkl
│   ├── methi_cluster_model.pkl
│   └── methi_scaler.pkl
│
├── Data
│   ├── stage1_predictions.csv
│   ├── stage2_accurate_clusters.csv
│   ├── stage3_habitability_scores.csv
│   └── all_75_habitable_exoplanets_scores.csv
│
└── Assets
```

---

## Machine Learning Models

METHI integrates multiple machine learning algorithms to improve prediction accuracy and robustness.

| Model | Purpose |
|--------|----------|
| Random Forest | Habitability classification |
| XGBoost | Gradient-boosted prediction |
| Multi-Layer Perceptron (MLP) | Neural network classification |
| Ensemble Model | Final prediction |
| K-Means | Clustering habitable candidates |

---

## Dataset

METHI uses publicly available data from the **NASA Exoplanet Archive**.

The primary astrophysical features include:

- Planet Radius
- Stellar Insolation Flux
- Stellar Effective Temperature
- Stellar Mass
- Stellar Radius

The dataset contains approximately **5,900 confirmed exoplanets**, from which METHI identifies and ranks potentially habitable worlds.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ruthwik-dhama/methi.git
cd methi
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the interactive Streamlit application:

```bash
streamlit run app.py
```

The web application allows users to:

- Predict exoplanet habitability
- Compare machine learning model predictions
- Explore habitability scores
- Visualize clustered candidate planets
- Rank potentially habitable exoplanets

---

## Results

Using approximately **5,900 confirmed exoplanets**, METHI successfully:

- Identified **75 high-potential habitable candidates**
- Generated continuous habitability scores through ensemble machine learning
- Clustered planets with similar astrophysical characteristics
- Produced an interpretable ranking of candidate exoplanets
- Delivered an interactive web platform for real-time prediction and exploration

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Streamlit
- Matplotlib

---

## Future Work

Planned improvements include:

- Atmospheric composition analysis
- Orbital eccentricity modeling
- Bayesian uncertainty estimation
- Explainable AI using SHAP values
- Transformer-based deep learning architectures
- Integration with future NASA Exoplanet Archive releases
- Incorporation of JWST observational data
- Automated model retraining for newly confirmed exoplanets

---

## Citation

If you use METHI in your research, please cite the published paper.

```bibtex
@inproceedings{dhama2025methi,
  author    = {Ruthwik Dhama and Praveen Pratap Singh and Vishal Kumar},
  title     = {METHI: An Ensemble-based Machine Learned Exoplanetary Habitability Index},
  booktitle = {2025 International Conference on New Media Studies (CONMEDIA)},
  year      = {2025},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/document/11290119}
}
```

If you use the software implementation, please also cite the GitHub repository.

```bibtex
@software{dhama2025methi,
  author = {Ruthwik Dhama},
  title  = {METHI: Machine-Learned Exoplanetary Habitability Index},
  year   = {2025},
  url    = {https://github.com/ruthwik-dhama/methi}
}
```

---

## Author

**Ruthwik Dhama**

Student Researcher in Astrophysics, Machine Learning, and Data Science

**Research Interests**

- Exoplanet Habitability
- Astroinformatics
- Machine Learning
- Stellar Astrophysics
- Scientific Computing

---

## Acknowledgments

This work was made possible through the support of the **Young Researchers Institute** and the use of publicly available datasets provided by the **NASA Exoplanet Archive**.

The author also acknowledges the contributions of **Praveen Pratap Singh** and **Vishal Kumar** for their mentorship and collaboration throughout the development of METHI.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
