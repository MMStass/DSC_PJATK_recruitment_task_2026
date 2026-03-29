# Loan Repayment Prediction
This repository contains the code and experiments for predicting loan repayment risk.
The project leverages advanced feature engineering, banking domain knowledge (e.g. custom scorecards, risk indicators), using Gradient Boosters (CatBoost, XGBoost, LightGBM).

# Project Structure

```
├── data/
│   ├── processed/         # Create this dierctory for processed .parquet files generated during FE
│   └── raw/               # Create this directory and place raw .csv files here.
├── logs/
│   ├── Optuna parameters/ # Saved hyperparameters from Optuna tuning
│   └── ...                # Experiment logs and feature importances (.csv, .json)
├── Notebooks/
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   ├── FE.ipynb           # Feature Engineering execution pipeline
│   └── ML.ipynb           # Machine Learning training, evaluation, and logging pipeline
├── References/
│   └── links.md           # Links to articles and resources used
└── src/
    ├── logging/
    │   └── logger.py      # Custom experiment logging system
    ├── optimization/      # Optuna optimization scripts
    └── utils/             # Core utility modules (cb_utils.py, xgb_utils.py, lgbm_utils.py, feature_utils.py, model_utils.py)
```

# How to Run the Project
## To reproduce the results, follow these steps in order:

### 1.Data Preparation
Place your initial raw data files into the data/raw/ directory and create the `data/preprocessed/` directory for FE output.

### 2.Feature Engineering
Open and execute all cells in `Notebooks/FE.ipynb`.
This notebook handles data cleaning, applies our custom transformations (including KNN feature borrowing, Target Encoding, and building the custom scorecard), and saves the optimized .parquet files into the `data/processed/` folder for faster loading.

### 3.Model Training and Prediction
Open and execute all cells in `Notebooks/ML.ipynb`.
This notebook contains the sequential, modular calls to our logic in the `src/ directory`. It trains the final CatBoost model and logs the experiment. To see the results, navigate to the logs/ directory.

# (PL) Dla organizatorów konkursu
link do prezentacji: https://docs.google.com/presentation/d/1WydZPS0neLyPusJbuK1YgyHxyAv9UC-qN3Ixd-cOh5k/edit?usp=sharing
