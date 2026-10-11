"""Persist and load trained model artifacts.

Three files make up a complete trained artifact bundle:
    model.pt     — PyTorch state_dict (the learned weights)
    scaler.pt    — feature mean/std used to standardize inputs
    columns.json — ordered list of feature names

All three are required to score new data: the scaler undoes training-time
standardization, and `columns.json` guarantees new rows are reordered to
match the exact layout the model was trained on.
"""

import json
from typing import Iterable

import numpy as np
import torch

from config import ARTIFACTS_DIR


def save_artifacts(
    model_state: dict,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    columns: Iterable[str],
) -> None:
    """Write model weights, scaler stats, and feature order to disk.

    Args:
        model_state:   Output of `model.state_dict()` for the chosen
                       checkpoint (typically the best-validation one).
        feature_mean:  Per-column means fit on the training split.
        feature_std:   Per-column stds fit on the training split.
        columns:       Feature names in the exact order the model expects.
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
