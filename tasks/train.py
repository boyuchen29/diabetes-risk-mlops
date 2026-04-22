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
from risk_score.train import run_training
from risk_score.mlflow_utils import log_run_metrics
from tasks.config_utils import load_runtime_config
from tasks.io_utils import dump_pickle, load_pickle


def run_train(dataset: tuple, config: dict, output_path: str, *, dbutils_client=None) -> dict:
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

    dump_pickle(train_output, output_path, dbutils=dbutils_client)

    return train_output


if __name__ == "__main__" or "__file__" not in globals():
    config = load_runtime_config(_root, dbutils_client=globals().get("dbutils"))

    run_id = dbutils.jobs.taskValues.get(taskKey="ingest", key="run_id")  # noqa: F821

    DATASET_PATH = config["pipeline"]["dataset_path"]
    TRAIN_OUTPUT_PATH = config["pipeline"]["train_output_path"]

    dataset = load_pickle(DATASET_PATH, dbutils=dbutils)  # noqa: F821

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_registry_uri("databricks")

    with mlflow.start_run(run_id=run_id):
        print("Running feature selection and training model...")
        print(config["feature_selection"]["best_subsets"])
        train_output = run_train(
            dataset,
            config,
            TRAIN_OUTPUT_PATH,
            dbutils_client=dbutils,  # noqa: F821
        )
        log_run_metrics(train_output["metrics"])
        print(f"AUC-ROC: {train_output['metrics']['auc']:.4f}")
        print(f"Train output saved to {TRAIN_OUTPUT_PATH}")
