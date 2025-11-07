import pandas as pd

SENSORS = [f"s{i}" for i in range(1,22)]

def add_cycle_norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cycle_norm"] = out["cycle"] / out.groupby("unit")["cycle"].transform("max")
    return out

def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_cycle_norm(df)
    # keep ops + sensors + cycle_norm
    keep = ["op1","op2","op3","cycle_norm"] + [c for c in df.columns if c in SENSORS]
    return df[["unit","cycle"] + keep + (["RUL"] if "RUL" in df.columns else [])]
