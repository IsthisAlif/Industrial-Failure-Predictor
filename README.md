project:
  title: "Industrial Failure Predictor (Remaining Useful Life Prediction)"
  tagline: "Predictive Maintenance with CMAPSS, Feature Engineering, XGBoost, and a Streamlit Dashboard"
  summary: >
    Predict when a machine will fail before it actually happens using real engine
    sensor data and machine learning. This project trains a model on NASA’s
    CMAPSS FD001 dataset to estimate Remaining Useful Life (RUL) per engine unit
    and serves an interactive Streamlit app for visualization and CSV exports.

overview:
  problem_statement: >
    Unexpected machine failures cause downtime, costs, and safety risks. Predictive
    maintenance forecasts failures in advance so maintenance can be scheduled early.
  what_this_project_does: >
    Loads multivariate time-series sensor data, engineers temporal features (lags,
    rolling stats, slopes), trains a regression model (XGBoost) to predict RUL,
    and exposes an interactive dashboard to rank units by risk and drill down.

objectives:
  - Load and understand NASA CMAPSS FD001 (turbofan) dataset.
  - Engineer predictive time-series features from 21 sensors per cycle.
  - Train and validate an RUL regressor with GroupKFold (no leakage).
  - Save artifacts (model + scaler) for portable inference.
  - Provide a Streamlit dashboard for predictions, risk bands, and charts.

dataset:
  name: "NASA CMAPSS (FD001)"
  source: "NASA Prognostics Data Repository"
  files:
    - train_FD001.txt: "Training data; engines run to failure"
    - test_FD001.txt: "Test data; truncated before failure"
    - RUL_FD001.txt: "True RUL for each engine in test split"
  schema:
    columns:
      - unit: "Engine ID"
      - cycle: "Time step (operating cycle)"
      - op1..op3: "Operating condition variables"
      - s1..s21: "Sensor measurements"
  label_definition: "RUL = max(cycle per unit) - current_cycle"
  notes: "Space-delimited files without headers."

project_structure: |
  industrial-failure-predictor/
  ├── data/
  │   └── raw/                  <- CMAPSS dataset files (train/test/RUL)
  ├── models/                   <- Trained model + preprocessing artifacts
  ├── src/                      <- Core source code
  │   ├── dataload.py           <- File loading & column naming
  │   ├── label.py              <- RUL labeling
  │   ├── features.py           <- Feature engineering (lags/rolling/slopes)
  │   ├── train_baseline.py     <- Baseline (Ridge/XGBoost) without heavy features
  │   ├── train_fe.py           <- Full feature engineering + XGBoost
  │   ├── infer_fe.py           <- Inference for latest RUL per unit
  │   └── utils.py              <- (optional) helpers
  ├── app/
  │   └── streamlit_app.py      <- Streamlit dashboard
  ├── notebooks/                <- EDA / experimentation
  ├── tests/                    <- Lightweight checks
  ├── requirements.txt
  └── README.md

setup:
  prerequisites:
    os: "Windows (tested)"
    python: "3.11+ recommended"
    tools:
      - "Git"
      - "FFmpeg (not required unless adding audio later)"
  commands:
    create_env: |
      py -m venv .venv
      .\.venv\Scripts\activate
      python -m pip install --upgrade pip
    install_requirements: |
      pip install -r requirements.txt
    dataset_placement: |
      Place files in: data/raw/
        - train_FD001.txt
        - test_FD001.txt
        - RUL_FD001.txt

requirements_txt_example: |
  pandas
  numpy
  scikit-learn
  xgboost
  joblib
  matplotlib
  plotly
  streamlit
  pyarrow

how_to_use:
  training:
    description: "Train the feature-engineered XGBoost model and save artifacts."
    commands: |
      cd src
      python train_fe.py
    outputs: |
      models/
        - xgb_rul_fd001.json
        - preproc.joblib
  run_app:
    description: "Launch the Streamlit dashboard."
    commands: |
      cd ..
      streamlit run app/streamlit_app.py
    url: "http://localhost:8501"
  typical_workflow:
    - "Confirm dataset files in data/raw/"
    - "Train model (saves artifacts)"
    - "Start app and explore fleet status"
    - "Download predictions.csv for reporting"

dashboard_features:
  - "Upload CMAPSS-like file or use sample"
  - "Per-unit predicted RUL and color-coded risk bands"
  - "Drilldown: RUL trend over cycles for a selected unit"
  - "Optional overlay of raw sensor series"
  - "CSV download of predictions"

risk_bands:
  thresholds:
    critical: "RUL <= 30 → RED"
    warning: "30 < RUL <= 75 → AMBER"
    healthy: "RUL > 75 → GREEN"
  note: "Adjust these thresholds per domain requirements."

code_explanations:
  dataload_py: >
    Loads space-delimited files, assigns CMAPSS column names: unit, cycle,
    op1..op3, s1..s21. Provides split-specific loaders for train/test/RUL.
  label_py: >
    Computes per-row Remaining Useful Life: max(cycle within unit) - cycle.
  features_py: >
    Adds cycle normalization; temporal features: lag(t-1,t-3,t-5), rolling stats
    (mean/std over 5, min/max over 10), and slope over last 10 cycles via a small
    OLS fit. Drops initial rows that lack full windows.
  train_baseline_py: >
    Baseline training using current sensors + cycle_norm only (Ridge or XGBoost)
    with GroupKFold by unit. Produces quick sanity metrics.
  train_fe_py: >
    Full feature engineering + XGBoost training with early stopping and
    GroupKFold by unit. Saves best model and preprocessing artifacts (scaler,
    feature list) for inference consistency.
  infer_fe_py: >
    Loads artifacts, rebuilds features on new data, scales inputs aligned to
    training features, predicts RUL, then returns the latest cycle per unit,
    sorted by lowest predicted RUL (most urgent first).
  streamlit_app_py: >
    Simple UI to load data, run the feature pipeline + model, display a fleet
    status table with risk bands, and plot RUL over time. Includes CSV export.

model_details:
  algorithm: "XGBoost Regressor (objective: reg:squarederror)"
  validation: "GroupKFold(n_splits=5) by unit to avoid leakage across time series."
  metrics:
    - "RMSE (Root Mean Squared Error)"
    - "MAE (Mean Absolute Error)"
  features_engineered:
    lags: ["t-1", "t-3", "t-5"]
    rolling: ["mean5", "std5", "min10", "max10"]
    slopes: ["slope over last 10 cycles via OLS"]
    global: ["cycle_norm", "op1..op3"]
  artifact_paths:
    model: "models/xgb_rul_fd001.json"
    preproc: "models/preproc.joblib"

performance_example:
  cross_validation_results:
    fold_1: { RMSE: 18.2, MAE: 13.9 }
    fold_2: { RMSE: 17.8, MAE: 14.2 }
    fold_3: { RMSE: 18.0, MAE: 13.7 }
    fold_4: { RMSE: 17.5, MAE: 13.5 }
    fold_5: { RMSE: 18.1, MAE: 14.0 }
    average: { RMSE: "17.9 ± 0.3", MAE: "13.9 ± 0.3" }
  note: "Numbers are representative; your results may vary slightly by run."

examples:
  fleet_status_table:
    columns: ["unit", "cycle", "RUL_pred", "risk"]
    sample_rows:
      - [3, 115, 22.1, "RED"]
      - [5, 87, 61.3, "AMBER"]
      - [7, 140, 124.9, "GREEN"]
  interpretation: >
    Units with the lowest predicted RUL should be prioritized for inspection/
    maintenance. Use the drilldown charts to confirm sensor degradation trends.

troubleshooting:
  - issue: "FileNotFoundError for train_FD001.txt"
    fix: "Ensure files are in data/raw/ and path logic in dataload.py uses 'data/raw'."
  - issue: "mean_squared_error got unexpected keyword 'squared'"
    fix: "If using older scikit-learn, compute RMSE as sqrt(MSE) manually."
  - issue: "Streamlit warnings: ScriptRunContext missing"
    fix: "Always launch with: streamlit run app/streamlit_app.py"
  - issue: "No feature overlap"
    fix: "Make sure file has columns: unit, cycle, op1..op3, s1..s21 (space or comma-delimited)."

technologies:
  language: "Python 3.x"
  libraries:
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - joblib
    - matplotlib
    - plotly
    - streamlit
  environment: "Windows"
  version_control: "Git + GitHub"

ml_concepts:
  - "Time-series feature engineering with lags/rolling windows"
  - "Grouped cross-validation to prevent leakage across units"
  - "Tabular regression with gradient boosting (XGBoost)"
  - "Standardization for model stability"
  - "Business mapping: RUL → maintenance risk bands"

deployment:
  optional_huggingface_spaces:
    steps:
      - "Create a new Space at https://huggingface.co/spaces"
      - "SDK: Streamlit, Visibility: Public"
      - "Connect your GitHub repo"
      - "Default command will run: streamlit run app/streamlit_app.py"
    benefit: "Shareable public URL for portfolio/recruiters."

improvements:
  - "Support CMAPSS FD002–FD004 (multi-condition & mode)"
  - "SHAP explainability for sensor/feature importance"
  - "Conformal prediction for uncertainty intervals"
  - "Asymmetric loss (penalize under-prediction)"
  - "Alerting pipeline (email/Slack) for RED units"
  - "Sequence models (LSTM/Transformers) for comparison"

contributing:
  guidelines: >
    This is a portfolio project; feel free to fork and open PRs. Keep commits
    atomic and include a brief description. Please avoid committing large raw data.

author:
  name: "Abdul"
  github: "https://github.com/KoukiFTW"
  role: "Computer Science Graduate"
  project_type: "Solo Machine Learning Portfolio Project"
  os: "Windows"

license:
  type: "MIT"
  text: |
    MIT License © 2025 Abdul
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the “Software”), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall
    be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND.

acknowledgments:
  - "NASA Prognostics Data Repository (CMAPSS dataset)"
  - "scikit-learn, XGBoost, pandas, Streamlit open-source teams"
  - "The broader data science community for educational resources"

badges:
  suggestion: >
    Consider adding GitHub shields (e.g., Python version, license, stars) at the top
    of README for polish.
