import numpy as np
import pandas as pd
import requests
from io import StringIO
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def build_dataset(config: dict) -> tuple:
    df_raw = _load_raw_data(config["data"])

    df = df_raw.copy()
    threshold = np.percentile(df["Y"], config["data"]["target_percentile"])
    df["Y"] = np.where(df["Y"] > threshold, 1, 0)

    df_categorized = _categorize_features(df)

    X_all = df_categorized.drop("y", axis=1)
    y_all = df_categorized["y"]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_all, y_all,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    X_train_balanced, y_train_balanced = _random_oversample(
        X_train_raw, y_train_raw, random_state=config["data"]["random_state"]
    )

    label_encoder = LabelEncoder()
    y_encoded_all = label_encoder.fit_transform(y_all)

    onehot = OneHotEncoder(sparse_output=False)
    onehot.fit(X_all)

    X_train_enc = onehot.transform(X_train_balanced)
    y_train_enc = label_encoder.transform(y_train_balanced)
    X_test_enc = onehot.transform(X_test_raw)
    y_test_enc = label_encoder.transform(y_test_raw)

    return (
        X_train_enc, X_test_enc,
        y_train_enc, y_test_enc,
        X_test_raw, X_train_raw,
        onehot, label_encoder,
        df_categorized, y_encoded_all,
    )


def _load_raw_data(data_config: dict) -> pd.DataFrame:
    data_path = data_config.get("path")
    if data_path:
        candidate_paths = _candidate_data_paths(data_path)
        for candidate in candidate_paths:
            if candidate.exists():
                return pd.read_csv(candidate, sep="\t")

    data_url = data_config.get("url")
    if not data_url:
        raise ValueError("data.path or data.url must be configured")

    response = requests.get(data_url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), sep="\t")


def _candidate_data_paths(data_path: str) -> list[Path]:
    path = Path(data_path)
    if path.is_absolute():
        return [path]

    repo_root = Path(__file__).resolve().parents[2]
    from_root = repo_root / path
    if from_root.resolve(strict=False) == path.resolve(strict=False):
        return [path]
    return [path, from_root]


def _random_oversample(
    X: pd.DataFrame, y: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    train_df = X.copy()
    train_df["y"] = y.to_numpy()

    class_counts = train_df["y"].value_counts()
    target_count = int(class_counts.max())

    balanced_parts = []
    for class_name, group in train_df.groupby("y"):
        replace = len(group) < target_count
        sampled_group = group.sample(
            n=target_count, replace=replace, random_state=random_state
        )
        balanced_parts.append(sampled_group)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    X_balanced = balanced.drop(columns="y")
    y_balanced = balanced["y"].rename("y")
    return X_balanced, y_balanced


CATEGORIZATION_SCHEMA = [
    {"feature": "age", "column": "AGE", "type": "cut",
     "bins": [-np.inf, 25, 35, 45, 55, np.inf],
     "labels": ["age_25", "age_35", "age_45", "age_55", "age_65"]},
    {"feature": "sex", "column": "SEX", "type": "threshold",
     "threshold": 1, "gt_label": "male", "le_label": "female"},
    {"feature": "bmi", "column": "BMI", "type": "cut",
     "bins": [-np.inf, 18.5, 25, 30, np.inf],
     "labels": ["underweight", "normal", "overweight", "obese"]},
    {"feature": "bp", "column": "BP", "type": "cut",
     "bins": [-np.inf, 80, 90, 120, np.inf],
     "labels": ["low", "normal", "elevated", "high"]},
    {"feature": "s1", "column": "S1", "type": "cut",
     "bins": [-np.inf, 150, 200, 240, np.inf],
     "labels": ["optimal", "normal", "borderline", "high"]},
    {"feature": "s2", "column": "S2", "type": "cut",
     "bins": [-np.inf, 100, 130, 160, np.inf],
     "labels": ["optimal", "normal", "borderline", "high"]},
    {"feature": "s3", "column": "S3", "type": "cut",
     "bins": [-np.inf, 40, 60, np.inf],
     "labels": ["low", "normal", "high"]},
    {"feature": "s4", "column": "S4", "type": "cut",
     "bins": [-np.inf, 3.5, 5, np.inf],
     "labels": ["optimal", "normal", "high"]},
    {"feature": "s5", "column": "S5", "type": "cut",
     "bins": [-np.inf, 4.3, 4.7, np.inf],
     "labels": ["low", "normal", "high"]},
    {"feature": "s6", "column": "S6", "type": "cut",
     "bins": [-np.inf, 90, 110, np.inf],
     "labels": ["normal", "prediabetes", "diabetes"]},
]


def categorize_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for spec in CATEGORIZATION_SCHEMA:
        col = df[spec["column"]]
        if spec["type"] == "cut":
            out[spec["feature"]] = pd.cut(col, bins=spec["bins"], labels=spec["labels"])
        else:
            out[spec["feature"]] = np.where(col > spec["threshold"], spec["gt_label"], spec["le_label"])
    return out


def _categorize_features(df: pd.DataFrame) -> pd.DataFrame:
    out = categorize_features(df)
    out["y"] = np.where(df["Y"] == 1, "worse", "better")
    return out
