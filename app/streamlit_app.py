# app/streamlit_app.py
from pathlib import Path
import sys
import pandas as pd
import joblib
import xgboost as xgb
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Resolve project paths ---
APP_DIR = Path(__file__).resolve().parent
PROJ_DIR = APP_DIR.parent
SRC_DIR = PROJ_DIR / "src"
MODELS_DIR = PROJ_DIR / "models"
DATA_DIR = PROJ_DIR / "data" / "raw"

# Ensure we can import from src/
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Local imports from your codebase
from dataload import load_fd_file
from features import build_features

st.set_page_config(page_title="Industrial Failure Predictor (RUL)", layout="wide")
st.title("🛠️ Industrial Failure Predictor (RUL)")
st.caption("CMAPSS FD001 • XGBoost + temporal features (lags, rolling stats, slopes)")

# --- Load model artifacts ---
preproc_path = MODELS_DIR / "preproc.joblib"
model_path = MODELS_DIR / "xgb_rul_fd001.json"

@st.cache_data(show_spinner=False)
def _artifact_exists():
    return preproc_path.exists() and model_path.exists()

artifacts_ok = _artifact_exists()
if not artifacts_ok:
    st.error(
        "Model artifacts not found. Train the FE model first:\n\n"
        "cd src\n"
        "python train_fe.py\n\n"
        "This should create models/preproc.joblib and models/xgb_rul_fd001.json"
    )
    st.stop()

# Load artifacts
meta = joblib.load(preproc_path)
scaler = meta["scaler"]
feature_cols = meta["features"]

model = xgb.Booster()
model.load_model(str(model_path))

# --- File input ---
st.subheader("1) Load data")
uploaded = st.file_uploader(
    "Upload CMAPSS-like file (CSV or space-delimited): columns = unit,cycle,op1..op3,s1..s21",
    type=["txt", "csv"],
)

if uploaded is not None:
    # Auto-detect comma or whitespace
    df = pd.read_csv(uploaded, sep=r"\s+|,", engine="python", header=None)
    st.success("File loaded from upload.")
else:
    fallback = DATA_DIR / "train_FD001.txt"
    if not fallback.exists():
        st.warning("No upload and no sample file at data/raw/train_FD001.txt. Please upload a file.")
        st.stop()
    df = pd.read_csv(fallback, sep=r"\s+", header=None)
    st.info("Using sample file: data/raw/train_FD001.txt")

# Assign column names based on CMAPSS FD001 schema
cols = ["unit","cycle"] + [f"op{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)]
df.columns = cols[: df.shape[1]]

# --- Feature pipeline & prediction ---
st.subheader("2) Predict Remaining Useful Life (RUL)")
with st.spinner("Building features and running model..."):
    df_feat = build_features(df).copy()

    use_cols = [c for c in feature_cols if c in df_feat.columns]
    if not use_cols:
        st.error("No feature overlap between data and model. Check file format (unit,cycle,op1..op3,s1..s21).")
        st.stop()

    X = scaler.transform(df_feat[use_cols].values)
    d = xgb.DMatrix(X)
    df_feat["RUL_pred"] = model.predict(d)

# Latest (current) cycle per unit
latest_idx = df_feat.groupby("unit")["cycle"].idxmax()
summary = df_feat.loc[latest_idx, ["unit", "cycle", "RUL_pred"]].sort_values("RUL_pred").reset_index(drop=True)

# Risk bands
summary["risk"] = pd.cut(
    summary["RUL_pred"],
    bins=[-1, 30, 75, 1e12],
    labels=["RED", "AMBER", "GREEN"],
)

# Display
st.subheader("3) Fleet status (lowest RUL first)")
st.dataframe(summary, use_container_width=True)

# Download
st.download_button(
    "⬇️ Download predictions.csv",
    data=summary.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv",
)

# --- Unit drilldown ---
st.subheader("4) Drilldown")
if len(summary) == 0:
    st.warning("No units found after feature generation. Check your file.")
    st.stop()

sel_unit = st.selectbox("Select a unit to inspect", summary["unit"].tolist())
u = df_feat[df_feat["unit"] == sel_unit].sort_values("cycle")

# RUL over time
st.markdown(f"**Predicted RUL over time — Unit {sel_unit}**")
st.line_chart(u.set_index("cycle")[["RUL_pred"]])

# Optional: allow user to plot a few raw sensors for context
sensor_choices = [c for c in df.columns if c.startswith("s")]
default_sensors = [s for s in ["s2","s3","s4"] if s in sensor_choices][:3]
picked = st.multiselect(
    "Overlay raw sensors (optional)",
    options=sensor_choices,
    default=default_sensors
)
if picked:
    st.line_chart(u.set_index("cycle")[picked])

# --- Test evaluation section  ---
st.subheader("5) Test Evaluation (FD001)")

test_results_path = MODELS_DIR / "test_results_fd001.csv"

if not test_results_path.exists():
    st.info(
        "Test evaluation file not found.\n\n"
        "Run this once in your terminal to generate it:\n"
        "  cd src\n"
        "  python test_evaluate.py"
    )
else:
    # Load test results produced by src/test_evaluate.py
    test_res = pd.read_csv(test_results_path)

    # Expecting columns: unit, RUL_pred, RUL_true, RUL_pred_adjusted
    if {"RUL_true", "RUL_pred_adjusted"}.issubset(test_res.columns):
        y_true = test_res["RUL_true"].values
        y_pred = test_res["RUL_pred_adjusted"].values

        rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
        mae = np.abs(y_true - y_pred).mean()

        st.markdown(f"**Test RMSE:** {rmse:.2f} &nbsp;&nbsp; **Test MAE:** {mae:.2f}")

        # Show a small table of first few engines
        st.markdown("**Sample of test engines (true vs predicted RUL):**")
        st.dataframe(test_res.head(10))

        # Scatter plot: Predicted vs True
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.7)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal (y = x)")
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL (adjusted)")
        ax.set_title("Predicted vs True RUL on Test FD001")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
    else:
        st.warning(
            "The test_results_fd001.csv file does not have the expected columns.\n"
            "Re-run: python src/test_evaluate.py"
        )