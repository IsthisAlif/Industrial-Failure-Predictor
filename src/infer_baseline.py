import os, pandas as pd, joblib
from xgboost import Booster, DMatrix
from dataload import load_fd001_split, load_fd_file
from features import build_baseline_features

def predict_file(path="data/raw/train_FD001.txt"):
    meta = joblib.load("models/baseline_preproc.joblib")
    scaler = meta["scaler"]; feature_cols = meta["features"]; model_type = meta["model_type"]

    df = load_fd_file(path)
    df_feat = build_baseline_features(df).dropna().reset_index(drop=True)

    X = scaler.transform(df_feat[feature_cols].values)

    if model_type == "ridge":
        model = joblib.load("models/baseline_ridge.joblib")
        pred = model.predict(X)
    else:
        model = Booster(); model.load_model("models/baseline_xgb.json")
        pred = model.predict(DMatrix(X))

    df_feat["RUL_pred"] = pred

    # take last cycle per unit
    idx = df_feat.groupby("unit")["cycle"].idxmax()
    out = df_feat.loc[idx, ["unit","cycle","RUL_pred"]].sort_values("RUL_pred").reset_index(drop=True)
    return out

if __name__ == "__main__":
    print(predict_file().head())
