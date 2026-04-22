import sys
import inspect
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

import mlflow
from risk_score.data import build_dataset
from risk_score.mlflow_utils import log_run_params
from tasks.config_utils import load_runtime_config
from tasks.io_utils import dump_pickle


def run_ingest(config: dict, output_path: str, *, dbutils_client=None) -> tuple:
    dataset = build_dataset(config)
    dump_pickle(dataset, output_path, dbutils=dbutils_client)
    return dataset


if __name__ == "__main__" or "__file__" not in globals():
    config = load_runtime_config(_root, dbutils_client=globals().get("dbutils"))

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_registry_uri("databricks")
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    _dbutils = globals().get("dbutils")
    DATASET_PATH = config["pipeline"]["dataset_path"]

    with mlflow.start_run() as run:
        log_run_params(config)
        print("Loading and preparing data...")
        run_ingest(config, DATASET_PATH, dbutils_client=_dbutils)
        print(f"Dataset saved to {DATASET_PATH}")
        _dbutils.jobs.taskValues.set(key="run_id", value=run.info.run_id)
        print(f"Run id: {run.info.run_id}")
