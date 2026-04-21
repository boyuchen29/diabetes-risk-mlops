import pytest
from risk_score.data import build_dataset
from risk_score.train import run_training

MANUAL_CONFIG = {
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

AUTO_CONFIG = {
    "data": MANUAL_CONFIG["data"],
    "feature_selection": {"mode": "auto", "best_subsets": []},
    "model": {"max_iter": 200, "random_state": 42},
}


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(MANUAL_CONFIG)


def test_manual_mode_uses_config_subsets(dataset):
    (X_train_enc, X_test_enc, y_train_enc, y_test_enc,
     _, _, onehot, _, df_categorized, _) = dataset
    feature_names = onehot.get_feature_names_out(
        df_categorized.drop("y", axis=1).columns
    )
    model, best_subsets, *_ = run_training(
        X_train_enc, X_test_enc, y_train_enc, y_test_enc,
        feature_names, df_categorized, MANUAL_CONFIG,
    )
    assert best_subsets == ["bp", "bmi", "s4", "s5"]


def test_auto_mode_returns_subsets(dataset):
    (X_train_enc, X_test_enc, y_train_enc, y_test_enc,
     _, _, onehot, _, df_categorized, _) = dataset
    feature_names = onehot.get_feature_names_out(
        df_categorized.drop("y", axis=1).columns
    )
    model, best_subsets, *_ = run_training(
        X_train_enc, X_test_enc, y_train_enc, y_test_enc,
        feature_names, df_categorized, AUTO_CONFIG,
    )
    valid_features = list(df_categorized.drop("y", axis=1).columns)
    assert len(best_subsets) > 0
    assert all(f in valid_features for f in best_subsets)


def test_model_returns_probabilities(dataset):
    (X_train_enc, X_test_enc, y_train_enc, y_test_enc,
     _, _, onehot, _, df_categorized, _) = dataset
    feature_names = onehot.get_feature_names_out(
        df_categorized.drop("y", axis=1).columns
    )
    model, best_subsets, onehot_best, X_train_enc2, X_test_enc2, *_ = run_training(
        X_train_enc, X_test_enc, y_train_enc, y_test_enc,
        feature_names, df_categorized, MANUAL_CONFIG,
    )
    probs = model.predict_proba(X_test_enc2)[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()
