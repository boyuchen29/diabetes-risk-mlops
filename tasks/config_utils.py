import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_runtime_config(repo_root: Path, *, dbutils_client=None, argv=None) -> dict:
    config_path = repo_root / "config.yaml"
    with open(config_path) as handle:
        config = yaml.safe_load(handle)

    parsed_args = _parse_args(argv)

    config["feature_selection"]["mode"] = _resolve_param(
        parsed_args.feature_selection_mode,
        "feature_selection.mode",
        config["feature_selection"]["mode"],
        dbutils_client=dbutils_client,
    )

    best_subsets_raw = _resolve_param(
        parsed_args.feature_selection_best_subsets,
        "feature_selection.best_subsets",
        ",".join(config["feature_selection"]["best_subsets"]),
        dbutils_client=dbutils_client,
    )
    config["feature_selection"]["best_subsets"] = [
        feature.strip() for feature in best_subsets_raw.split(",") if feature.strip()
    ]

    config["data"]["test_size"] = float(
        _resolve_param(
            parsed_args.data_test_size,
            "data.test_size",
            config["data"]["test_size"],
            dbutils_client=dbutils_client,
        )
    )
    config["data"]["random_state"] = int(
        _resolve_param(
            parsed_args.data_random_state,
            "data.random_state",
            config["data"]["random_state"],
            dbutils_client=dbutils_client,
        )
    )
    config["model"]["max_iter"] = int(
        _resolve_param(
            parsed_args.model_max_iter,
            "model.max_iter",
            config["model"]["max_iter"],
            dbutils_client=dbutils_client,
        )
    )
    config["model"]["random_state"] = int(
        _resolve_param(
            parsed_args.model_random_state,
            "model.random_state",
            config["model"]["random_state"],
            dbutils_client=dbutils_client,
        )
    )

    return config


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--feature-selection-mode")
    parser.add_argument("--feature-selection-best-subsets")
    parser.add_argument("--data-test-size")
    parser.add_argument("--data-random-state")
    parser.add_argument("--model-max-iter")
    parser.add_argument("--model-random-state")
    parsed_args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return parsed_args


def _resolve_param(cli_value, widget_key: str, default, *, dbutils_client=None) -> str:
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value).strip()

    if dbutils_client is not None:
        try:
            widget_value = dbutils_client.widgets.get(widget_key)
            if widget_value.strip():
                return widget_value.strip()
        except Exception as exc:
            logger.debug("Databricks widget %r not available: %s", widget_key, exc)

    return str(default)
