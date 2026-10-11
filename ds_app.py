"""Streamlit interface for Titanic survival inference.

Workflow:
    1. User points to an inference CSV (unlabeled Titanic-shaped file).
    2. App loads the trained artifacts from `./artifacts/`.
    3. App runs the same preprocessing pipeline the model was trained on.
    4. App shows predictions and offers a CSV download.

Run with:
    streamlit run ds_app.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

from artifacts import load_artifacts
from model import TitanicMLP
from preprocess import preprocess_inference

st.set_page_config(page_title="Titanic Survival Classifier", layout="wide")
st.title("Titanic Survival Classifier")
st.caption("Load a Titanic Test CSV, load the trained model, inspect predictions.")


# ---------------------------- Sidebar ----------------------------
st.sidebar.header("Inference input")
csv_path = st.sidebar.text_input(
    "Path to inference CSV",
    value="./data/test.csv",
)
run = st.sidebar.button("Run inference", type="primary")

st.sidebar.divider()
if st.sidebar.button("Shutdown server", help="Stops the Streamlit process."):
    st.sidebar.success("Shutting down…")
    os._exit(0)


# ---------------------------- Helpers ----------------------------
@st.cache_resource
def load_model():
    """Load trained artifacts and build a ready-to-use model."""
    model_state, feature_mean, feature_std, columns, imputers = load_artifacts()
    model = TitanicMLP(input_features=len(columns))
    model.load_state_dict(model_state)
    model.eval()
    return model, feature_mean, feature_std, columns, imputers


def predict(model, X_scaled: np.ndarray):
    """Return `(predicted_labels, predicted_probabilities)` as NumPy arrays."""
    with torch.no_grad():
        logits = model(torch.from_numpy(X_scaled.astype(np.float32)))
        probabilities = torch.sigmoid(logits).numpy()
    labels = (probabilities >= 0.5).astype(int)
    return labels, probabilities


def plot_probability_histogram(y_prob):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(y_prob, bins=30, edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="threshold")
    ax.set_xlabel("P(Survived)")
    ax.set_ylabel("Passengers")
    ax.set_title("Model confidence distribution")
    ax.legend(loc="upper center")
    return fig


def plot_predicted_counts(y_pred):
    counts = (
        pd.Series(y_pred)
        .map({0: "Did not survive", 1: "Survived"})
        .value_counts()
        .reindex(["Did not survive", "Survived"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_title("Predicted outcomes")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    return fig


def plot_rate_by_feature(df, y_pred, feature):
    plot_df = pd.DataFrame({feature: df[feature].values, "Predicted": y_pred})
    rate = plot_df.groupby(feature, observed=True)["Predicted"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=rate, x=feature, y="Predicted", hue=feature, legend=False, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted survival rate")
    ax.set_title(f"Survival rate by {feature}")
    return fig


def plot_rate_by_age(df, y_pred):
    buckets = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 55, 200],
        labels=["0-12", "13-18", "19-35", "36-55", "55+"],
    )
    plot_df = pd.DataFrame({"Age group": buckets, "Predicted": y_pred}).dropna(
        subset=["Age group"]
    )
    rate = plot_df.groupby("Age group", observed=True)["Predicted"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=rate, x="Age group", y="Predicted", hue="Age group", legend=False, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted survival rate")
    ax.set_title("Survival rate by Age group")
    return fig


# ---------------------------- Main ----------------------------
if not run:
    st.info("Enter a CSV path in the sidebar and click **Run inference**.")
    st.stop()

csv_file = Path(csv_path)
if not csv_file.exists():
    st.error(f"File not found: `{csv_file}`")
    st.stop()

inference_df = pd.read_csv(csv_file)

st.subheader("Dataset preview")
st.caption(f"{len(inference_df)} rows")
st.dataframe(inference_df.head(10), use_container_width=True)

# Load model + saved imputers, preprocess with no re-fit, scale, predict.
model, feature_mean, feature_std, columns, imputers = load_model()
X_inference = preprocess_inference(inference_df, imputers, columns)
X_scaled = (X_inference.to_numpy(dtype=np.float32) - feature_mean) / feature_std
predicted_labels, predicted_probabilities = predict(model, X_scaled)

# ------------------ Predictions table ------------------
st.subheader("Predictions")
predictions_df = pd.DataFrame({
    "PassengerId": inference_df.get("PassengerId", pd.Series(range(len(inference_df)))),
    "Predicted": predicted_labels,
    "P(Survived)": predicted_probabilities.round(4),
})
st.dataframe(predictions_df, use_container_width=True, height=320)

st.download_button(
    "Download predictions as CSV",
    data=predictions_df.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)

# ------------------ Prediction insights ------------------
st.subheader("Prediction insights")

top_col1, top_col2 = st.columns(2)
with top_col1:
    st.pyplot(plot_probability_histogram(predicted_probabilities))
with top_col2:
    st.pyplot(plot_predicted_counts(predicted_labels))

bottom_col1, bottom_col2, bottom_col3 = st.columns(3)
with bottom_col1:
    st.pyplot(plot_rate_by_feature(inference_df, predicted_labels, "Sex"))
with bottom_col2:
    st.pyplot(plot_rate_by_feature(inference_df, predicted_labels, "Pclass"))
with bottom_col3:
    st.pyplot(plot_rate_by_age(inference_df, predicted_labels))