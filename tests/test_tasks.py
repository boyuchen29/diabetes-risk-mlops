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
    run_ingest(CONFIG, output_path)
    assert Path(output_path).exists()
    with open(output_path, "rb") as f:
        dataset = pickle.load(f)
    assert len(dataset) == 10
