<h1 align="center">🏭 Industrial Failure Predictor</h1>
<h3 align="center">🔧 Remaining Useful Life (RUL) Prediction using CMAPSS, XGBoost & Streamlit</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" alt="Framework"/>
  <img src="https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost" alt="Model"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <a href="https://github.com/KoukiFTW"><img src="https://img.shields.io/badge/Author-Abdul-black" alt="Author"/></a>
</p>

---

## 🧩 Overview

**Industrial Failure Predictor** forecasts when an industrial machine will fail *before it happens* using NASA’s **CMAPSS FD001 turbofan dataset**.  
It uses advanced **feature engineering** and **machine learning (XGBoost)** to estimate **Remaining Useful Life (RUL)** and visualizes the results in a sleek **Streamlit dashboard**.

### 🎯 Problem
Unexpected failures cause costly downtime and safety risks.  
Predictive maintenance forecasts failures ahead of time — so you can plan maintenance *before breakdowns occur.*

### ⚙️ Solution
- Process real engine sensor data (time-series)
- Engineer predictive features (lags, rolling stats, slopes)
- Train a regression model (XGBoost) to estimate RUL
- Visualize predictions in an interactive Streamlit dashboard

---

## 📊 Project Demo

> 🎮 Launch the app locally:
```bash
streamlit run app/streamlit_app.py
💻 Or deploy publicly on Hugging Face Spaces
(SDK: Streamlit → Connect your GitHub → Auto-launches)

📦 Dataset: NASA CMAPSS FD001
File	Description
train_FD001.txt	Engine runs to failure (training set)
test_FD001.txt	Truncated runs before failure (test set)
RUL_FD001.txt	True RUL per engine (ground truth)

🧾 Label Definition:

RUL = max(cycle_per_unit) - current_cycle

🧠 Schema Example

sql
Copy code
unit, cycle, op1, op2, op3, s1 ... s21
📁 Project Structure
graphql
Copy code
Industrial-Failure-Predictor/
├── data/
│   └── raw/                  # CMAPSS dataset files
├── models/                   # Trained model + scaler
├── src/                      # Core scripts
│   ├── dataload.py           # Load and name CMAPSS files
│   ├── label.py              # Compute Remaining Useful Life
│   ├── features.py           # Feature engineering (lags, rolling, slopes)
│   ├── train_fe.py           # Full feature + XGBoost training
│   ├── infer_fe.py           # Predict RUL on new data
│   └── utils.py              # Helpers
├── app/
│   └── streamlit_app.py      # Interactive dashboard
├── notebooks/                # EDA & experiments
├── requirements.txt
└── README.md
🧠 Model Details
Component	Description
Algorithm	XGBoost Regressor (reg:squarederror)
Validation	GroupKFold (5 splits per engine unit)
Metrics	RMSE, MAE
Features	Lag (t-1, t-3, t-5), Rolling (mean/std/min/max), Slopes (10-cycle OLS), op1–op3, cycle_norm
Artifacts	xgb_rul_fd001.json, preproc.joblib

📈 Sample Results
Fold	RMSE	MAE
1	18.2	13.9
2	17.8	14.2
3	18.0	13.7
4	17.5	13.5
5	18.1	14.0
Avg	17.9 ± 0.3	13.9 ± 0.3

💻 How to Run
1️⃣ Create Virtual Environment
bash
Copy code
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
Example requirements.txt:

nginx
Copy code
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
plotly
streamlit
pyarrow
3️⃣ Place Dataset
bash
Copy code
data/raw/
  ├── train_FD001.txt
  ├── test_FD001.txt
  └── RUL_FD001.txt
4️⃣ Train Model
bash
Copy code
cd src
python train_fe.py
5️⃣ Launch Dashboard
bash
Copy code
cd ..
streamlit run app/streamlit_app.py
🖥️ Opens at: http://localhost:8501

📉 Dashboard Features
✨ Upload CMAPSS-like dataset or use sample
✨ Predict RUL for each engine (sorted by risk)
✨ Visualize degradation curves over cycles
✨ Color-coded risk bands
✨ Download predictions as CSV

Risk	Rule	Color
Critical	RUL ≤ 30	🔴 Red
Warning	30 < RUL ≤ 75	🟠 Amber
Healthy	RUL > 75	🟢 Green

🧮 Example Output
unit	cycle	RUL_pred	risk
3	115	22.1	🔴 Critical
5	87	61.3	🟠 Warning
7	140	124.9	🟢 Healthy

Engines with lowest predicted RUL → highest maintenance priority.

🧱 Technologies
Category	Tools
Language	Python 3.11+
Libraries	pandas, numpy, scikit-learn, xgboost, streamlit, plotly, joblib
ML Concepts	Time-series feature engineering, Grouped CV, Gradient Boosting
Environment	Windows
Version Control	Git + GitHub

🧩 Troubleshooting
Issue	Fix
FileNotFoundError	Ensure files are in data/raw/
mean_squared_error() got unexpected keyword 'squared'	Update scikit-learn
Streamlit warnings	Always use streamlit run app/streamlit_app.py
“No feature overlap”	Check your columns: unit, cycle, op1..op3, s1..s21

🧭 Roadmap / Future Improvements
 Extend to CMAPSS FD002–FD004 (multi-mode)

 SHAP explainability for sensor importance

 Conformal prediction (uncertainty bounds)

 Asymmetric loss for under-prediction penalty

 Alerting (Slack/email) for critical engines

 Compare with sequence models (LSTM / Transformers)

🤝 Contributing
Contributions are welcome!
Please fork, make changes, and open a Pull Request.
Keep commits atomic and avoid committing large datasets.

👤 Author
Abdul
🎓 Computer Science Graduate
🌐 GitHub Profile
🖥️ Windows environment | Solo ML portfolio project

⚖️ License
MIT License © 2025 Abdul

sql
Copy code
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND.
🙏 Acknowledgments
NASA Prognostics Data Repository (CMAPSS)

scikit-learn, XGBoost, pandas, Streamlit teams

The open-source data science community 🌍

