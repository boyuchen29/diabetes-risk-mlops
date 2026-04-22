from pathlib import Path

from tasks.config_utils import load_runtime_config


def test_load_runtime_config_applies_cli_overrides():
    repo_root = Path(__file__).resolve().parents[1]

    config = load_runtime_config(
        repo_root,
        argv=[
            "--feature-selection-mode", "manual",
            "--feature-selection-best-subsets", "bp,bmi,s1,s5",
            "--data-test-size", "0.3",
            "--data-random-state", "7",
            "--model-max-iter", "400",
            "--model-random-state", "99",
        ],
    )

    assert config["feature_selection"]["mode"] == "manual"
    assert config["feature_selection"]["best_subsets"] == ["bp", "bmi", "s1", "s5"]
    assert config["data"]["test_size"] == 0.3
    assert config["data"]["random_state"] == 7
    assert config["model"]["max_iter"] == 400
    assert config["model"]["random_state"] == 99
