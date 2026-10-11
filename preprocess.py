"""Feature engineering and cleaning for the Titanic dataset.

The pipeline exposes a single `preprocess(train_df, test_df)` entry point
that returns model-ready NumPy-friendly DataFrames. Design choices:

* **No leakage.** All imputation statistics (median Age by Pclass+Sex,
  median Fare by Pclass, modal Embarked) are learned from the training
  set only, then applied to both splits. If we fit on the full dataset
  we would smuggle information about the test rows back into training.

* **Feature engineering over raw columns.** `Name`, `Ticket`, and `Cabin`
  carry strong signal but need extraction (`Title`, `TicketGroup`,
  `Deck`) before a model can use them. The raw columns are dropped afterward.

* **Column alignment.** Test data is reindexed to match the training
  feature matrix so that one-hot encoding never produces mismatched
  columns across splits (e.g. a Deck letter that only appears in test).
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
        # Age varies strongly with class and sex (1st-class women are
        # older on average than 3rd-class men), so a group-wise median
        # is more accurate than a global median.
        "age_by_class_sex": train.groupby(["Pclass", "Sex"])["Age"].median(),
        # Fare depends almost entirely on Pclass.
        "fare_by_class": train.groupby("Pclass")["Fare"].median(),
        # Embarked has just two missing values; filling with the mode
        # ('S', Southampton) is safe.
        "embarked_mode": train["Embarked"].mode().iloc[0],
    }


def _impute(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    """Fill missing Age / Fare / Embarked using the pre-computed stats."""
    df = df.copy()

    # Look up the right median for each row's (Pclass, Sex) combo. We
    # build a Series aligned to df.index because `MultiIndex.map` would
    # return an Index, which `fillna` won't accept.
    age_lookup = imputers["age_by_class_sex"]
    age_fill = pd.Series(
        [age_lookup.get((p, s)) for p, s in zip(df["Pclass"], df["Sex"])],
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


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Transform raw Kaggle CSVs into model-ready feature matrices.

    Args:
        train_df: The raw `train.csv` DataFrame (must contain `Survived`).
        test_df:  The raw `test.csv` DataFrame (no `Survived` column).

    Returns:
        A tuple `(X_train, y_train, X_test)` where:
            X_train — DataFrame of engineered/encoded features, no NaNs.
            y_train — Series of binary survival labels (0 or 1).
            X_test  — DataFrame with the exact same columns as X_train,
                      in the same order; new/missing one-hot columns are
                      filled with 0.

    Imputation statistics are learned from `train_df` only to prevent
    leakage from the test set into training.
    """
    y_train = train_df["Survived"].copy()
    train = train_df.drop(columns=["Survived"])

    train = _add_features(train)
    test = _add_features(test_df)

    imputers = _fit_imputers(train)
    train = _impute(train, imputers)
    test = _impute(test, imputers)

    train = _encode(train).drop(columns=DROP_COLS)
    test = _encode(test).drop(columns=DROP_COLS)

    # Guarantee identical column schemas across splits. A Deck letter
    # (or any rare category) present in only one split would otherwise
    # produce mismatched one-hot columns.
    test = test.reindex(columns=train.columns, fill_value=0)

    summarize(train, y_train, test)
    return train, y_train, test


def summarize(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> None:
    """Print a human-readable summary of what preprocessing produced.

    Highlights the shapes, confirms there are no missing values left,
    shows the class balance, lists the engineered feature groups, and
    previews the first few rows — enough to spot-check that the
    pipeline did what we expected.
    """
    engineered = ["Title", "FamilySize", "IsAlone", "Deck", "TicketGroup"]
    one_hot_groups = {
        "Embarked": [c for c in X_train.columns if c.startswith("Embarked_")],
        "Title":    [c for c in X_train.columns if c.startswith("Title_")],
        "Deck":     [c for c in X_train.columns if c.startswith("Deck_")],
    }
    survived = int(y_train.sum())
    total = len(y_train)

    print("============ Preprocessing summary ============")
    print(f"Shapes       : X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}")
    print(f"Missing left : X_train={X_train.isna().sum().sum()}, X_test={X_test.isna().sum().sum()}")
    print(f"Class balance: {survived}/{total} survived ({survived / total:.1%})")
    print(f"Dropped raw  : {DROP_COLS}")
    print(f"Engineered   : {engineered}")
    for base, cols in one_hot_groups.items():
        print(f"One-hot {base:<8}: {cols}")

    print("\n------------ X_train.head() ------------")
    print(X_train.head())
    print("================================================\n")
