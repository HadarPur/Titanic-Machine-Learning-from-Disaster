"""Download the Titanic competition dataset from Kaggle.

Writes `train.csv`, `test.csv`, and `gender_submission.csv` into the
local `data/` directory. Running the script requires a Kaggle API token
at `~/.kaggle/kaggle.json` and acceptance of the competition rules on
kaggle.com (otherwise the download returns a 403).

Run once after cloning the repo; the rest of the pipeline reads from
`./data/`.
"""

import os
import kagglehub

# Create the data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download the Titanic competition dataset into the data folder.
# `force_download=True` re-fetches even if a cached copy exists, so the
# script is safe to re-run after Kaggle updates the files.
path = kagglehub.competition_download(
    "titanic",
    output_dir="data",
    force_download=True
)

print(f"Dataset downloaded to {path} directory")
