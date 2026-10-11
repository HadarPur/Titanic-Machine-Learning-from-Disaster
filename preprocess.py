"""Feature engineering and cleaning for the Titanic dataset.

Two entry points:

    preprocess(train_df)
        Fits imputation stats and returns `(X_train, y_train, imputers)`.
        Used at training time.

    preprocess_inference(df, imputers, columns)
        Applies the same transformations to new rows using previously
        fitted imputers, and reindexes the result to match the training
        feature schema. Used at inference time (e.g. the Streamlit app).

Design choices:

* **No leakage.** Imputation statistics (median Age by Pclass+Title,
  median Fare by Pclass, modal Embarked) are learned from the training
  set only. Inference reuses the saved stats — it never re-fits.

* **Feature engineering over raw columns.** `Name`, `Ticket`, and `Cabin`
  carry strong signal but need extraction (`Title`, `TicketGroup`,
  `Deck`) before a model can use them. The raw columns are dropped afterward.

* **Column alignment.** Inference data is reindexed to the training
  column schema so one-hot encoding never produces mismatched columns
  (e.g. a Deck letter that only appears in inference).
"""

import numpy as np
import pandas as pd

# Collapse the long tail of titles (Dr, Rev, Col, Lady, ...) into the
# five that actually have survival signal. Mme/Mlle/Ms are French/old
# variants of Mrs/Miss. Everything else becomes "Rare".
TITLE_MAP = {
    "Mr": "Mr",
    "Mrs": "Mrs", "Mme": "Mrs", "Ms": "Mrs",
    "Miss": "Miss", "Mlle": "Miss",
    "Master": "Master",
}

# Columns dropped after feature engineering:
#   - PassengerId: identifier, no signal
#   - Name:        signal extracted into Title
#   - Ticket:      signal extracted into TicketGroup
#   - Cabin:       signal extracted into Deck
#   - SibSp/Parch: combined into FamilySize + IsAlone
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]


def _extract_title(name: str) -> str:
    """Pull the honorific out of a Titanic-style name string.

    Names look like `"Braund, Mr. Owen Harris"` — the title sits between
    the comma and the period. Unknown titles fall through to "Rare".
    """
    raw = name.split(",")[1].split(".")[0].strip()
    return TITLE_MAP.get(raw, "Rare")


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from raw columns.

    Added columns:
        Title       — Mr / Mrs / Miss / Master / Rare (strong survival signal)
        FamilySize  — SibSp + Parch + 1 (the passenger themselves)
        IsAlone     — 1 if FamilySize == 1, else 0
        Deck        — first letter of Cabin, 'U' if missing
        TicketGroup — number of passengers sharing the same ticket number
                      (families and groups often shared tickets)
    """
    df = df.copy()
    df["Title"] = df["Name"].map(_extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df["TicketGroup"] = df.groupby("Ticket")["Ticket"].transform("count")
    return df


def _fit_imputers(train: pd.DataFrame) -> dict:
    """Compute the imputation statistics used to fill missing values.

    Learned exclusively from the training split — see the module
    docstring on leakage.
    """
    return {
        # Age is best predicted by Title (Master => boy, Miss => young
        # woman, Mr/Mrs => adult, Rare => older) combined with Pclass.
        # Grouping by Title instead of Sex correctly handles boys with
        # the "Master" title — they'd otherwise be filled with the adult
        # male median.
        "age_by_class_title": train.groupby(["Pclass", "Title"])["Age"].median(),
        # Fare depends almost entirely on Pclass.
        "fare_by_class": train.groupby("Pclass")["Fare"].median(),
        # Embarked has just two missing values; filling with the mode
        # ('S', Southampton) is safe.
        "embarked_mode": train["Embarked"].mode().iloc[0],
    }


def _impute(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    """Fill missing Age / Fare / Embarked using the pre-computed stats."""
    df = df.copy()

    # Look up the right median for each row's (Pclass, Title) combo.
    # We build a Series aligned to df.index because `MultiIndex.map`
    # would return an Index, which `fillna` won't accept.
    age_lookup = imputers["age_by_class_title"]
    age_fill = pd.Series(
        [age_lookup.get((p, t)) for p, t in zip(df["Pclass"], df["Title"])],
        index=df.index,
    )
    df["Age"] = df["Age"].fillna(age_fill)
    df["Fare"] = df["Fare"].fillna(df["Pclass"].map(imputers["fare_by_class"]))
    df["Embarked"] = df["Embarked"].fillna(imputers["embarked_mode"])
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns into numeric form a NN can consume.

    * Sex       — binary (female=1, male=0)
    * Embarked  — one-hot (3 categories)
    * Title     — one-hot (5 categories)
    * Deck      — one-hot (8 categories + U for unknown)
    * Pclass    — left as ordinal 1/2/3 since the ordering is meaningful
    """
    df = df.copy()
    df["Sex"] = (df["Sex"] == "female").astype(int)
    df = pd.get_dummies(df, columns=["Embarked", "Title", "Deck"], dtype=int)
    return df


def preprocess(train_df: pd.DataFrame):
    """Fit the training-time pipeline on `train_df`.

    Args:
        train_df: The raw `train.csv` DataFrame (must contain `Survived`).

    Returns:
        A tuple `(X_train, y_train, imputers)` where:
            X_train  — DataFrame of engineered/encoded features, no NaNs.
            y_train  — Series of binary survival labels (0 or 1).
            imputers — Fitted imputation stats; persist these alongside
                       the model so `preprocess_inference` can reuse them.
    """
    y_train = train_df["Survived"].copy()
    train = train_df.drop(columns=["Survived"])

    train = _add_features(train)
    imputers = _fit_imputers(train)
    train = _impute(train, imputers)
    train = _encode(train).drop(columns=DROP_COLS)

    summarize(train, y_train)
    return train, y_train, imputers


def preprocess_inference(df: pd.DataFrame, imputers: dict, columns) -> pd.DataFrame:
    """Apply the training-time pipeline to new rows using saved imputers.

    Args:
        df:       Raw inference DataFrame (no `Survived` column needed).
        imputers: Imputation stats returned by `preprocess`.
        columns:  The training feature schema (X_train.columns) — the
                  output is reindexed to this exact column list so new
                  or missing one-hot categories are handled consistently.

    Returns:
        DataFrame with the same columns (and order) as the training set.
    """
    df = _add_features(df)
    df = _impute(df, imputers)
    df = _encode(df).drop(columns=DROP_COLS, errors="ignore")
    return df.reindex(columns=columns, fill_value=0)


def summarize(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Print a human-readable summary of what preprocessing produced.

    Highlights the shape, confirms there are no missing values left,
    shows the class balance, lists the engineered feature groups, and
    previews the first few rows — enough to spot-check that the
    pipeline did what we expected.
    """
    imputed_and_kept = {
        "Age":      "median by (Pclass, Title)",
        "Fare":     "median by Pclass",
        "Embarked": "mode ('S')",
    }
    engineered = ["Title", "FamilySize", "IsAlone", "Deck", "TicketGroup"]
    one_hot_groups = {
        "Embarked": [c for c in X_train.columns if c.startswith("Embarked_")],
        "Title":    [c for c in X_train.columns if c.startswith("Title_")],
        "Deck":     [c for c in X_train.columns if c.startswith("Deck_")],
    }
    survived = int(y_train.sum())
    total = len(y_train)

    print("============ Preprocessing summary ============")
    print(f"Shapes       : X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Missing left : X_train={X_train.isna().sum().sum()}")
    print(f"Class balance: {survived}/{total} survived ({survived / total:.1%})")
    print(f"Dropped raw  : {DROP_COLS}")
    print("Imputed+kept :")
    for col, strategy in imputed_and_kept.items():
        print(f"  - {col:<9} <- {strategy}")
    print(f"Engineered   : {engineered}")
    for base, cols in one_hot_groups.items():
        print(f"One-hot {base:<8}: {cols}")

    print("\n------------ X_train.head() ------------")
    print(X_train.head())
    print("================================================\n")
