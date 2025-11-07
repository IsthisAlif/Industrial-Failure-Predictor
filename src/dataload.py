import os
import pandas as pd

COLS = ["unit","cycle"] + [f"op{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)]

def load_fd_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS, engine="python")

def load_fd001_split(root="data/raw", split="train"):
    if split == "train":
        fn = "train_FD001.txt"
        df = load_fd_file(os.path.join(root, fn))
    elif split == "test":
        fn = "test_FD001.txt"
        df = load_fd_file(os.path.join(root, fn))
    elif split == "rul":
        fn = "RUL_FD001.txt"
        df = pd.read_csv(os.path.join(root, fn), sep=r"\s+", header=None, names=["RUL"])
    else:
        raise ValueError("split must be train|test|rul")
    return df
