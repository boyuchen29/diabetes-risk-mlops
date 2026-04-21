import pytest
import mlflow
from risk_score.mlflow_utils import (
    _flatten_config,
    log_run_params,
    log_run_metrics,
    log_and_register_model,
)


def test_flatten_config_dot_notation():
    flat = _flatten_config({
        "data": {"test_size": 0.2, "random_state": 42},
        "feature_selection": {"best_subsets": ["bp", "bmi"]},
        "mlflow": {"enabled": True},
    })
    assert flat["data.test_size"] == 0.2
    assert flat["data.random_state"] == 42
    assert flat["mlflow.enabled"] is True
    assert flat["feature_selection.best_subsets"] == '["bp", "bmi"]'


def test_log_run_params_writes_flattened_keys(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-params")
    config = {"data": {"test_size": 0.2, "random_state": 42}}

    with mlflow.start_run() as run:
        log_run_params(config)

    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run.info.run_id).data.params
    assert params["data.test_size"] == "0.2"
    assert params["data.random_state"] == "42"


def test_log_run_metrics_skips_non_scalar(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-metrics")

    with mlflow.start_run() as run:
        log_run_metrics({"auc": 0.85, "confusion_matrix": [[10, 2], [3, 8]]})

    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run.info.run_id).data.metrics
    assert "auc" in metrics
    assert metrics["auc"] == pytest.approx(0.85)
    assert "confusion_matrix" not in metrics


def test_log_and_register_model_returns_run_id(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-model")

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)

    with mlflow.start_run():
        run_id = log_and_register_model(
            model, artifact_path="model", model_name="test-model", X_sample=X
        )

    assert isinstance(run_id, str) and len(run_id) > 0
