"""Central configuration for the Titanic training pipeline.

All hyperparameters and output paths live here so that `train.py` and any
downstream script (e.g. a predict / Streamlit app) read the exact same
values. Change a knob in one place and the rest of the project follows.
"""

from pathlib import Path

# Random seed used for NumPy, PyTorch, and the stratified train/val split.
# Fixing it makes training reproducible run-to-run.
SEED = 42

# Fraction of the training set held out for validation (early stopping,
# checkpoint selection). 0.3 keeps ~267 passengers for validation — a
# larger holdout reduces the noise in the val-accuracy signal (one
# correct/wrong flip is ~0.4pp instead of ~0.6pp at 0.2) at the cost
# of ~90 fewer training rows.
VAL_FRACTION = 0.3

# Mini-batch size for SGD updates. 64 is a good default on CPU for ~600
# training rows: large enough for stable gradients, small enough that we
# still get ~10 updates per epoch.
BATCH_SIZE = 64

# Adam learning rate. 1e-3 is the canonical starting point for MLPs on
# tabular data; small enough to converge smoothly, large enough to finish
# in a few seconds.
LEARNING_RATE = 1e-3

# L2 regularization on the weights. Modest value (1e-4) nudges the model
# away from overfitting without materially slowing convergence.
WEIGHT_DECAY = 1e-4

# Hard upper bound on training epochs. Early stopping will almost always
# fire first, but this guarantees the script terminates.
MAX_EPOCHS = 200

# Number of consecutive epochs without a new best validation accuracy
# before training is cut short.
EARLY_STOP_PATIENCE = 20

# Directory where trained weights, the feature scaler, and the column
# order are persisted after a successful run.
ARTIFACTS_DIR = Path("artifacts")
