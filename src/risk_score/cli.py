import argparse
import sys

import yaml

from risk_score.data import build_dataset
from risk_score.train import run_training
from risk_score.score import build_risk_scores


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_train(args):
    config = _load_config(args.config)

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

    print("Computing risk scores...")
    scores, weights, R = build_risk_scores(
        model, df_categorized, best_subsets, onehot_best, y_encoded_all, config
    )

    cm = metrics["confusion_matrix"]
    print("\n--- Results ---")
    print(f"Selected features : {best_subsets}")
    print(f"AUC-ROC           : {metrics['auc']:.4f}")
    print(f"True Negative     : {cm[0][0]}")
    print(f"False Positive    : {cm[0][1]}")
    print(f"False Negative    : {cm[1][0]}")
    print(f"True Positive     : {cm[1][1]}")
    print(f"Risk score range  : {R['RS'].min():.1f} - {R['RS'].max():.1f}")
    print(f"Feature weights   : {dict(zip(best_subsets, [round(float(w), 4) for w in weights]))}")


def cmd_predict(args):
    print("predict command not yet implemented — coming in the serving sub-project.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="risk-score")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run full training pipeline")
    train_parser.add_argument("--config", required=True, help="Path to config.yaml")

    predict_parser = subparsers.add_parser("predict", help="Score new patients (coming soon)")
    predict_parser.add_argument("--config", required=True, help="Path to config.yaml")
    predict_parser.add_argument("--input", required=True, help="Path to input JSON")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
