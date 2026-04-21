import sys
import inspect
from pathlib import Path


def _script_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent

    frame = inspect.currentframe()
    if frame is None or frame.f_code.co_filename is None:
        raise RuntimeError("Unable to determine script location")

    return Path(frame.f_code.co_filename).resolve().parent


_root = _script_dir()
sys.path.insert(0, str(_root / "src"))

import yaml
import mlflow

from risk_score.data import build_dataset
from risk_score.train import run_training
from risk_score.score import build_risk_scores
from risk_score.mlflow_utils import log_run_params, log_run_metrics, log_and_register_model, log_scoring_artifacts

CONFIG_PATH = _root / "config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

mlflow_cfg = config["mlflow"]
mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
mlflow.set_registry_uri("databricks")
mlflow.set_experiment(mlflow_cfg["experiment_name"])

with mlflow.start_run():
    log_run_params(config)

    print("Loading and preparing data...")
    (X_train_enc, X_test_enc, y_train_enc, y_test_enc,
     X_test_raw, X_train_raw, onehot, label_encoder,
     df_categorized, y_encoded_all) = build_dataset(config)

    feature_names = onehot.get_feature_names_out(
        df_categorized.drop("y", axis=1).columns
    )

    print("Running feature selection and training model...")
    (model, best_subsets, onehot_best,
     X_train_enc2, X_test_enc2,
     y_train_enc2, y_test_enc2,
     metrics) = run_training(
        X_train_enc, X_test_enc, y_train_enc, y_test_enc,
        feature_names, df_categorized, config,
    )

    log_run_metrics(metrics)

    print("Computing risk scores...")
    scores, weights, R = build_risk_scores(
        model, df_categorized, best_subsets, onehot_best, y_encoded_all, config
    )

    log_scoring_artifacts(scores, weights)

    run_id = log_and_register_model(
        model,
        name="model",
        model_name=mlflow_cfg["model_name"],
        X_sample=X_test_enc2,
    )

    print(f"Run id            : {run_id}")
    print(f"Selected features : {best_subsets}")
    print(f"AUC-ROC           : {metrics['auc']:.4f}")
    print(f"Risk score range  : {R['RS'].min():.1f} - {R['RS'].max():.1f}")
