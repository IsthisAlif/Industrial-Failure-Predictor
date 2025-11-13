from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt

# =====================================================
# PATH SETUP
# =====================================================
APP_DIR = Path(__file__).resolve().parent
PROJ_DIR = APP_DIR.parent
SRC_DIR = PROJ_DIR / "src"
MODELS_DIR = PROJ_DIR / "models"
DATA_DIR = PROJ_DIR / "data" / "raw"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from features import build_features

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Machine Health Dashboard",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Machine Health Dashboard")
st.write(
    """
This dashboard estimates **how much life is left** in each engine based on its sensor data.

It is designed to be **simple, clear, and easy to understand**:

- Each row represents **one engine**  
- The app estimates **remaining life (in cycles)**  
- Engines are classified into **green, yellow, and red** based on health  
"""
)

st.markdown("---")

# =====================================================
# LOAD MODEL
# =====================================================
preproc_path = MODELS_DIR / "preproc.joblib"
model_path = MODELS_DIR / "xgb_rul_fd001.json"

if not preproc_path.exists() or not model_path.exists():
    st.error(
        "Model files not found.\n"
        "Please train the model first:\n\n"
        "`cd src`\n`python train_fe.py`"
    )
    st.stop()

meta = joblib.load(preproc_path)
scaler = meta["scaler"]
feature_cols = meta["features"]

model = xgb.Booster()
model.load_model(str(model_path))

# =====================================================
# 1) DATA INPUT
# =====================================================
st.header("1. Upload Engine Data")

st.write(
    """
Upload engine sensor data in CSV or TXT format.  
If no file is uploaded, the dashboard uses a built-in example dataset.
"""
)

uploaded = st.file_uploader("Upload a sensor data file", type=["csv", "txt"])

if uploaded:
    df = pd.read_csv(uploaded, sep=r"\s+|,", engine="python", header=None)
    st.success("File uploaded successfully.")
else:
    example_path = DATA_DIR / "test_FD001.txt"
    df = pd.read_csv(example_path, sep=r"\s+", header=None)
    st.info("Using example dataset (test_FD001.txt).")

# Assign column names
cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
df.columns = cols[:df.shape[1]]

with st.expander("Show raw data"):
    st.dataframe(df.head(50), height=250, use_container_width=True)

# =====================================================
# 2) PREDICT REMAINING LIFE
# =====================================================
st.header("2. Remaining Life Prediction")

with st.spinner("Analysing data..."):
    df_feat = build_features(df)

    valid_cols = [c for c in feature_cols if c in df_feat.columns]
    X = scaler.transform(df_feat[valid_cols])
    df_feat["RUL_pred"] = model.predict(xgb.DMatrix(X))

# Take the latest cycle per engine
latest = df_feat.groupby("unit")["cycle"].idxmax()
summary = df_feat.loc[latest, ["unit", "cycle", "RUL_pred"]].sort_values("RUL_pred")

summary["risk"] = pd.cut(
    summary["RUL_pred"],
    bins=[-1, 30, 75, 1e9],
    labels=["🔴 CRITICAL", "🟠 WARNING", "🟢 HEALTHY"]
)

# Human-friendly names
display = summary.rename(
    columns={
        "unit": "Engine ID",
        "cycle": "Last Recorded Cycle",
        "RUL_pred": "Estimated Remaining Life (cycles)",
        "risk": "Health Status"
    }
)

st.write(
    """
The table below shows the **latest health estimate** for each engine:

- **Remaining Life (cycles)** = how long the engine may run before failing  
- **Health Status** = color-coded risk level  
"""
)

st.dataframe(display, height=300, use_container_width=True)

# =====================================================
# 3) ENGINE-LEVEL VIEW
# =====================================================
st.header("3. Inspect a Specific Engine")

engine_list = display["Engine ID"].tolist()
chosen_engine = st.selectbox("Select an engine:", engine_list)

engine_df = df_feat[df_feat["unit"] == chosen_engine].sort_values("cycle")

st.write("### Remaining Life Over Time")
st.line_chart(
    engine_df.set_index("cycle")[["RUL_pred"]].rename(
        columns={"RUL_pred": "Estimated Remaining Life"}
    )
)

available_sensors = [c for c in df.columns if c.startswith("s")]
default_sensors = ["s2", "s3", "s4"]

picked = st.multiselect(
    "Show sensor trends:",
    options=available_sensors,
    default=[s for s in default_sensors if s in available_sensors]
)

if picked:
    st.write("### Sensor Trends")
    st.line_chart(engine_df.set_index("cycle")[picked])

# =====================================================
# 4) DOWNLOAD SUMMARY
# =====================================================
st.header("4. Download Summary Report")

st.download_button(
    "📥 Download Engine Summary (CSV)",
    data=display.to_csv(index=False),
    file_name="engine_summary.csv",
    mime="text/csv"
)

# =====================================================
# 5) OPTIONAL: TECHNICAL MODEL ACCURACY
# =====================================================
st.header("5. Technical: Model Accuracy (Optional)")

st.write(
    """
This section shows how accurate the model is using a separate test dataset  
that includes **true actual values** of remaining engine life.
"""
)

results_path = MODELS_DIR / "test_results_fd001.csv"

if not results_path.exists():
    st.info(
        "Test evaluation file not found.\n"
        "To generate it:\n\n"
        "`cd src`\n`python test_evaluate.py`"
    )
else:
    test_res = pd.read_csv(results_path)

    y_true = test_res["RUL_true"]
    y_pred = test_res["RUL_pred_adjusted"]

    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    mae = np.mean(np.abs(y_true - y_pred))

    st.write(f"**Average Error (RMSE):** {rmse:.2f} cycles")
    st.write(f"**Average Absolute Error (MAE):** {mae:.2f} cycles")

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([0, max(y_true)], [0, max(y_true)], "r--")
    ax.set_xlabel("True Remaining Life")
    ax.set_ylabel("Predicted Remaining Life")
    ax.set_title("Predicted vs True Remaining Life")
    ax.grid(True)
    st.pyplot(fig)

st.caption("Simple, clear, and practical machine health dashboard. Built with Streamlit.")
