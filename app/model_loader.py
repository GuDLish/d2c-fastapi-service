"""Loads the artifact dict produced by Part 3 (churn_model.ipynb)."""
import os, joblib
from functools import lru_cache

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

@lru_cache(maxsize=1)
def get_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Place model.pkl from Part 3 in the project root, or run `python train_model.py`."
        )
    art = joblib.load(MODEL_PATH)
    required = {"pipeline", "threshold", "feature_cols", "num_cols", "cat_cols"}
    missing = required - set(art.keys())
    if missing:
        raise ValueError(f"Artifact missing keys: {missing}")
    return art
