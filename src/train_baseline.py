import os, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from xgboost import DMatrix, train as xgb_train
from dataload import load_fd001_split
from label import add_rul_labels
from features import build_baseline_features

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_eval(model_type="ridge"):
    # 1) Load & label
    df = load_fd001_split(split="train")
    df = add_rul_labels(df)
    df = build_baseline_features(df).dropna().reset_index(drop=True)

    # 2) Build matrices
    feature_cols = [c for c in df.columns if c not in ["unit","cycle","RUL"]]
    X = df[feature_cols].values
    y = df["RUL"].values
    groups = df["unit"].values

    # 3) Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 4) Grouped CV
    gkf = GroupKFold(n_splits=5)
    rmses, maes = [], []
    fold_models = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        if model_type == "ridge":
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)

        elif model_type == "xgb":
            dtr = DMatrix(Xtr, label=ytr)
            dva = DMatrix(Xva, label=yva)
            params = dict(objective="reg:squarederror", max_depth=6, eta=0.1,
                          subsample=0.9, colsample_bytree=0.9, nthread=0)
            model = xgb_train(params, dtr, num_boost_round=400,
                              evals=[(dva,"val")], early_stopping_rounds=30, verbose_eval=False)
            pred = model.predict(dva)
        else:
            raise ValueError("model_type must be 'ridge' or 'xgb'")

        mse  = mean_squared_error(yva, pred)
        rmse = mse ** 0.5
        mae  = mean_absolute_error(yva, pred)
        print(f"Fold {fold} — RMSE={rmse:.2f}  MAE={mae:.2f}")

        rmses.append(rmse); maes.append(mae)
        fold_models.append(model)

    print(f"\nCV RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
    print(f"CV  MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")

    # 5) Save best model + preproc
    best = int(np.argmin(rmses))
    best_model = fold_models[best]

    if model_type == "ridge":
        joblib.dump(best_model, os.path.join(MODELS_DIR, "baseline_ridge.joblib"))
        model_path = os.path.join(MODELS_DIR, "baseline_ridge.joblib")
    else:
        best_model.save_model(os.path.join(MODELS_DIR, "baseline_xgb.json"))
        model_path = os.path.join(MODELS_DIR, "baseline_xgb.json")

    joblib.dump({"scaler": scaler, "features": feature_cols, "model_type": model_type},
                os.path.join(MODELS_DIR, "baseline_preproc.joblib"))

    print(f"Saved model to: {model_path}")
    print("Saved preproc to: models/baseline_preproc.joblib")

if __name__ == "__main__":
    # try ridge first (fast), switch to 'xgb' after
    train_and_eval(model_type="ridge")
