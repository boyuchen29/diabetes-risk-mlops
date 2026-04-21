import numpy as np
import pandas as pd
import pytest
from risk_score.data import build_dataset, _categorize_features

CONFIG = {
    "data": {
        "path": "data/diabetes.tab.txt",
        "url": "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt",
        "target_percentile": 75,
        "test_size": 0.2,
        "random_state": 42,
    },
    "feature_selection": {"mode": "manual", "best_subsets": ["bp", "bmi", "s4", "s5"]},
    "model": {"max_iter": 200, "random_state": 42},
}


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(CONFIG)


def test_binarize_target(dataset):
    *_, y_encoded_all = dataset
    worse_ratio = (y_encoded_all == 1).mean()
    assert 0.20 <= worse_ratio <= 0.30, f"Expected ~25% worse, got {worse_ratio:.2f}"


def test_categorize_features():
    row = {
        "AGE": 40, "SEX": 2, "BMI": 27, "BP": 85,
        "S1": 180, "S2": 120, "S3": 50, "S4": 4.0,
        "S5": 4.5, "S6": 95, "Y": 0,
    }
    df = pd.DataFrame([row])
    result = _categorize_features(df)
    assert result["age"].iloc[0] == "age_45"
    assert result["sex"].iloc[0] == "male"
    assert result["bmi"].iloc[0] == "overweight"
    assert result["bp"].iloc[0] == "normal"
    assert result["y"].iloc[0] == "better"


def test_split_before_oversample(dataset):
    (X_train_enc, X_test_enc, y_train_enc, y_test_enc,
     X_test_raw, X_train_raw, onehot, label_encoder,
     df_categorized, y_encoded_all) = dataset
    total = len(df_categorized)
    expected_test = int(total * 0.2)
    assert abs(len(X_test_raw) - expected_test) <= 2
    _, counts_train = np.unique(y_train_enc, return_counts=True)
    assert counts_train[0] == counts_train[1], "Train classes should be balanced after oversampling"
