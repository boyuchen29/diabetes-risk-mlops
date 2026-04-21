import numpy as np
import pytest
from risk_score.data import build_dataset
from risk_score.train import run_training
from risk_score.score import build_risk_scores

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
def pipeline_output():
    (X_train_enc, X_test_enc, y_train_enc, y_test_enc,
     X_test_raw, X_train_raw, onehot, label_encoder,
     df_categorized, y_encoded_all) = build_dataset(CONFIG)
    feature_names = onehot.get_feature_names_out(
        df_categorized.drop("y", axis=1).columns
    )
    model, best_subsets, onehot_best, *_ = run_training(
        X_train_enc, X_test_enc, y_train_enc, y_test_enc,
        feature_names, df_categorized, CONFIG,
    )
    scores, weights, R = build_risk_scores(
        model, df_categorized, best_subsets, onehot_best, y_encoded_all, CONFIG
    )
    return scores, weights, R


def test_scores_in_range(pipeline_output):
    scores, _, _ = pipeline_output
    for feature, feature_scores in scores.items():
        for level, score in feature_scores.items():
            assert 0 <= score <= 1000, f"{feature}.{level} score={score} out of [0, 1000]"


def test_weights_sum_to_one(pipeline_output):
    _, weights, _ = pipeline_output
    assert abs(sum(weights) - 1.0) < 1e-6, f"Weights sum to {sum(weights)}"


def test_risk_scores_in_range(pipeline_output):
    _, _, R = pipeline_output
    assert (R["RS"] >= 0).all() and (R["RS"] <= 100).all(), \
        f"RS range: {R['RS'].min():.2f} - {R['RS'].max():.2f}"
