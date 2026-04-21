import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn


def log_run_params(config: dict) -> None:
    mlflow.log_params(_flatten_config(config))


def log_run_metrics(metrics: dict) -> None:
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if scalar_metrics:
        mlflow.log_metrics(scalar_metrics)


def log_and_register_model(model, *, name: str, model_name: str, X_sample) -> str:
    mlflow.sklearn.log_model(model, name=name, input_example=X_sample)
    run_id = mlflow.active_run().info.run_id
    mlflow.register_model(f"runs:/{run_id}/{name}", model_name)
    return run_id


def log_scoring_artifacts(scores: dict, weights) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "scores.json").write_text(json.dumps(scores))
        (tmp_path / "weights.json").write_text(json.dumps(list(weights)))
        mlflow.log_artifact(str(tmp_path / "scores.json"))
        mlflow.log_artifact(str(tmp_path / "weights.json"))


def _flatten_config(config: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in config.items():
        nested_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, nested_key))
        elif isinstance(value, (str, int, float, bool)):
            flat[nested_key] = value
        else:
            flat[nested_key] = json.dumps(value, sort_keys=True)
    return flat
