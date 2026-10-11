import numpy as np
import pandas as pd

TITLE_MAP = {
    "Mr": "Mr",
    "Mrs": "Mrs", "Mme": "Mrs", "Ms": "Mrs",
    "Miss": "Miss", "Mlle": "Miss",
    "Master": "Master",
}

DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]


def _extract_title(name: str) -> str:
    raw = name.split(",")[1].split(".")[0].strip()
    return TITLE_MAP.get(raw, "Rare")


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].map(_extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df["TicketGroup"] = df.groupby("Ticket")["Ticket"].transform("count")
    return df


def _fit_imputers(train: pd.DataFrame) -> dict:
    return {
        "age_by_class_sex": train.groupby(["Pclass", "Sex"])["Age"].median(),
        "fare_by_class": train.groupby("Pclass")["Fare"].median(),
        "embarked_mode": train["Embarked"].mode().iloc[0],
    }


def _impute(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    df = df.copy()
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
    df = df.copy()
    df["Sex"] = (df["Sex"] == "female").astype(int)
    df = pd.get_dummies(df, columns=["Embarked", "Title", "Deck"], dtype=int)
    return df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Return (X_train, y_train, X_test) with aligned columns.

    Imputation statistics are learned from `train_df` only to prevent leakage.
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

    test = test.reindex(columns=train.columns, fill_value=0)
    return train, y_train, test
