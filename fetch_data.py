import os
import kagglehub

# Create the data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download the Titanic competition dataset into the data folder
path = kagglehub.competition_download(
    "titanic",
    output_dir="data",
    force_download=True
)

print(f"Dataset downloaded to {path} directory")