import pickle
import pytest
from pathlib import Path

CONFIG = {
    "data": {
        "path": "data/diabetes.tab.txt",
        "url": "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt",
        "target_percentile": 75,
        "test_size": 0.2,
        "random_state": 42,
    },
    "feature_selection": {"mode": "manual", "best_subsets": ["bp", "bmi", "s4", "s5"]},
    "model": {"max_iter": 200, "random_state": 42},
}


def test_ingest_saves_dataset(tmp_path):
    from tasks.ingest import run_ingest
    output_path = str(tmp_path / "dataset.pkl")
    dataset_returned = run_ingest(CONFIG, output_path)
    assert Path(output_path).exists()
    assert len(dataset_returned) == 10
    with open(output_path, "rb") as f:
        dataset_pickled = pickle.load(f)
    assert len(dataset_pickled) == 10


def test_train_produces_model_and_metrics(tmp_path):
    import pickle
    from tasks.ingest import run_ingest
    from tasks.train import run_train

    dataset_path = str(tmp_path / "dataset.pkl")
    train_output_path = str(tmp_path / "train_output.pkl")

    dataset = run_ingest(CONFIG, dataset_path)

    train_output = run_train(dataset, CONFIG, train_output_path)

    assert Path(train_output_path).exists()
    assert "model" in train_output
    assert "metrics" in train_output
    assert 0.0 <= train_output["metrics"]["auc"] <= 1.0
    assert "best_subsets" in train_output
    assert len(train_output["best_subsets"]) > 0
    assert "onehot_best" in train_output
    assert "X_test_enc2" in train_output

    with open(train_output_path, "rb") as f:
        train_output_pickled = pickle.load(f)
    assert set(train_output_pickled.keys()) == {"model", "best_subsets", "onehot_best", "X_test_enc2", "metrics"}
