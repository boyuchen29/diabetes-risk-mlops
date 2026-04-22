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
from risk_score.train import run_training
from risk_score.mlflow_utils import log_run_metrics


def _get_param(key: str, default):
    try:
        val = dbutils.widgets.get(key)  # noqa: F821
        return val.strip() if val.strip() else str(default)
    except Exception:
        return str(default)


def run_train(dataset: tuple, config: dict, output_path: str) -> dict:
    (X_train_enc, X_test_enc, y_train_enc, y_test_enc,
     X_test_raw, X_train_raw, onehot, label_encoder,
     df_categorized, y_encoded_all) = dataset

    feature_names = onehot.get_feature_names_out(
        df_categorized.drop("y", axis=1).columns
    )

    (model, best_subsets, onehot_best,
     _, X_test_enc2,
     _, _,
     metrics) = run_training(
        X_train_enc, X_test_enc, y_train_enc, y_test_enc,
        feature_names, df_categorized, config,
    )

    train_output = {
        "model": model,
        "best_subsets": best_subsets,
        "onehot_best": onehot_best,
        "X_test_enc2": X_test_enc2,
        "metrics": metrics,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(train_output, f)

    return train_output


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

    run_id = dbutils.jobs.taskValues.get(taskKey="ingest", key="run_id")  # noqa: F821

    DATASET_PATH = "/dbfs/tmp/diabetes-risk/pipeline_data/dataset.pkl"
    TRAIN_OUTPUT_PATH = "/dbfs/tmp/diabetes-risk/pipeline_data/train_output.pkl"

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_registry_uri("databricks")

    with mlflow.start_run(run_id=run_id):
        print("Running feature selection and training model...")
        train_output = run_train(dataset, config, TRAIN_OUTPUT_PATH)
        log_run_metrics(train_output["metrics"])
        print(f"AUC-ROC: {train_output['metrics']['auc']:.4f}")
        print(f"Train output saved to {TRAIN_OUTPUT_PATH}")
