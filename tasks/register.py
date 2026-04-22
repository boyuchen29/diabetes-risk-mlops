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

import yaml
import mlflow
from risk_score.score import build_risk_scores
from risk_score.mlflow_utils import log_scoring_artifacts, log_and_register_model
from tasks.io_utils import load_pickle


def _get_param(key: str, default):
    try:
        val = dbutils.widgets.get(key)  # noqa: F821
        return val.strip() if val.strip() else str(default)
    except Exception:
        return str(default)


def run_register(dataset: tuple, train_output: dict, config: dict) -> tuple:
    (_, _, _, _, _, _, _, _, df_categorized, y_encoded_all) = dataset
    model = train_output["model"]
    best_subsets = train_output["best_subsets"]
    onehot_best = train_output["onehot_best"]

    scores, weights, R = build_risk_scores(
        model, df_categorized, best_subsets, onehot_best, y_encoded_all, config
    )
    return scores, weights, R


if __name__ == "__main__" or "__file__" not in globals():
    CONFIG_PATH = _root / "config.yaml"
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config["data"]["test_size"] = float(_get_param("data.test_size", config["data"]["test_size"]))
    config["data"]["random_state"] = int(_get_param("data.random_state", config["data"]["random_state"]))
    config["model"]["max_iter"] = int(_get_param("model.max_iter", config["model"]["max_iter"]))
    config["model"]["random_state"] = int(_get_param("model.random_state", config["model"]["random_state"]))

    run_id = dbutils.jobs.taskValues.get(taskKey="ingest", key="run_id")  # noqa: F821

    DATASET_PATH = "/dbfs/tmp/diabetes-risk/pipeline_data/dataset.pkl"
    TRAIN_OUTPUT_PATH = "/dbfs/tmp/diabetes-risk/pipeline_data/train_output.pkl"

    dataset = load_pickle(DATASET_PATH, dbutils=dbutils)  # noqa: F821
    train_output = load_pickle(TRAIN_OUTPUT_PATH, dbutils=dbutils)  # noqa: F821

    model = train_output["model"]
    best_subsets = train_output["best_subsets"]
    X_test_enc2 = train_output["X_test_enc2"]
    metrics = train_output["metrics"]

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_registry_uri("databricks")

    with mlflow.start_run(run_id=run_id):
        print("Computing risk scores...")
        scores, weights, R = run_register(dataset, train_output, config)
        log_scoring_artifacts(scores, weights, best_subsets)

        run_id_out = log_and_register_model(
            model,
            name="model",
            model_name=mlflow_cfg["model_name"],
            X_sample=X_test_enc2,
        )

        print(f"Run id            : {run_id_out}")
        print(f"Selected features : {best_subsets}")
        print(f"AUC-ROC           : {metrics['auc']:.4f}")
        print(f"Risk score range  : {R['RS'].min():.1f} - {R['RS'].max():.1f}")
