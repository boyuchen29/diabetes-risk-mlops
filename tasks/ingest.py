import sys
import inspect
import pickle
from pathlib import Path


def _repo_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    frame = inspect.currentframe()
    if frame is None or frame.f_code.co_filename is None:
        raise RuntimeError("Unable to determine script location")
    return Path(frame.f_code.co_filename).resolve().parent.parent


_root = _repo_root()
sys.path.insert(0, str(_root / "src"))

import yaml
import mlflow
from risk_score.data import build_dataset
from risk_score.mlflow_utils import log_run_params


def _get_param(key: str, default):
    try:
        val = dbutils.widgets.get(key)  # noqa: F821
        return val.strip() if val.strip() else str(default)
    except Exception:
        return str(default)


def run_ingest(config: dict, output_path: str) -> tuple:
    dataset = build_dataset(config)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


if __name__ == "__main__" or "__file__" not in globals():
    CONFIG_PATH = _root / "config.yaml"
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config["feature_selection"]["mode"] = _get_param(
        "feature_selection.mode", config["feature_selection"]["mode"]
    )
    best_subsets_raw = _get_param(
        "feature_selection.best_subsets",
        ",".join(config["feature_selection"]["best_subsets"]),
    )
    config["feature_selection"]["best_subsets"] = [f.strip() for f in best_subsets_raw.split(",")]
    config["data"]["test_size"] = float(_get_param("data.test_size", config["data"]["test_size"]))
    config["data"]["random_state"] = int(_get_param("data.random_state", config["data"]["random_state"]))
    config["model"]["max_iter"] = int(_get_param("model.max_iter", config["model"]["max_iter"]))
    config["model"]["random_state"] = int(_get_param("model.random_state", config["model"]["random_state"]))

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_registry_uri("databricks")
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    DATASET_PATH = "/dbfs/tmp/diabetes-risk/pipeline_data/dataset.pkl"

    with mlflow.start_run() as run:
        log_run_params(config)
        print("Loading and preparing data...")
        run_ingest(config, DATASET_PATH)
        print(f"Dataset saved to {DATASET_PATH}")
        dbutils.jobs.taskValues.set(key="run_id", value=run.info.run_id)  # noqa: F821
        print(f"Run id: {run.info.run_id}")
