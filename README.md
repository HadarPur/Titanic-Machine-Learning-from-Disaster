# Titanic - Machine Learning from Disaster

A PyTorch solution to the classic [Kaggle Titanic competition](https://www.kaggle.com/c/titanic). An MLP classifier predicts whether a passenger survived based on features like class, sex, age, and ticket details.

## Project structure

| File | Purpose |
|---|---|
| `fetch_data.py` | Downloads the competition dataset from Kaggle into `data/` |
| `titanic_eda.ipynb` | Exploratory data analysis of the training set |
| `preprocess.py` | Missing-value imputation, feature engineering, encoding |
| `model.py` | `TitanicMLP` — a 2-hidden-layer PyTorch classifier |
| `config.py` | Hyperparameters and artifact paths |
| `train.py` | Trains the model and saves weights to `artifacts/` |
| `ds_app.py` | Streamlit app for interactive predictions |

## Setup

```bash
git clone https://github.com/HadarPur/Titanic-Machine-Learning-from-Disaster.git
cd Titanic-Machine-Learning-from-Disaster

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Download the dataset (requires a Kaggle API token in `~/.kaggle/kaggle.json`):

```bash
python fetch_data.py
```

Explore the data:

```bash
jupyter notebook titanic_eda.ipynb
```

Train the model — writes `model.pt`, `scaler.pt`, and `columns.json` into `artifacts/`:

```bash
python train.py
```

Launch the interactive app:

```bash
streamlit run ds_app.py
```