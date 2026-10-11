"""Persist and load trained model artifacts.

Four files make up a complete trained artifact bundle:
    model.pt     — PyTorch state_dict (the learned weights)
    scaler.pt    — feature mean/std used to standardize inputs
    columns.json — ordered list of feature names
    imputers.pkl — fitted imputation stats (Age/Fare/Embarked)

All four are required to score new data: the scaler undoes training-time
standardization, `columns.json` guarantees new rows are reordered to
match the exact layout the model was trained on, and `imputers.pkl`
lets inference fill missing values without re-reading `train.csv`.
"""

import json
import pickle
from typing import Iterable

import numpy as np
import torch

from config import ARTIFACTS_DIR


def save_artifacts(
    model_state: dict,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    columns: Iterable[str],
    imputers: dict,
) -> None:
    """Write model weights, scaler stats, feature order, and imputers to disk.

    Args:
        model_state:   Output of `model.state_dict()` for the chosen
                       checkpoint (typically the best-validation one).
        feature_mean:  Per-column means fit on the training split.
        feature_std:   Per-column stds fit on the training split.
        columns:       Feature names in the exact order the model expects.
        imputers:      Fitted imputation stats from `preprocess()`.
    """
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    torch.save(model_state, ARTIFACTS_DIR / "model.pt")
    torch.save(
        {
            "mean": torch.from_numpy(feature_mean),
            "std": torch.from_numpy(feature_std),
        },
        ARTIFACTS_DIR / "scaler.pt",
    )
    with open(ARTIFACTS_DIR / "columns.json", "w") as columns_file:
        json.dump(list(columns), columns_file)
    with open(ARTIFACTS_DIR / "imputers.pkl", "wb") as imputers_file:
        pickle.dump(imputers, imputers_file)


def load_artifacts():
    """Load a previously saved artifact bundle from disk.

    Returns:
        A tuple `(model_state, feature_mean, feature_std, columns, imputers)`.
    """
    model_state = torch.load(ARTIFACTS_DIR / "model.pt", map_location="cpu")
    scaler = torch.load(ARTIFACTS_DIR / "scaler.pt", map_location="cpu")
    feature_mean = scaler["mean"].numpy()
    feature_std = scaler["std"].numpy()
    with open(ARTIFACTS_DIR / "columns.json") as columns_file:
        columns = json.load(columns_file)
    with open(ARTIFACTS_DIR / "imputers.pkl", "rb") as imputers_file:
        imputers = pickle.load(imputers_file)
    return model_state, feature_mean, feature_std, columns, imputers
