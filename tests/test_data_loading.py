from pathlib import Path

import pandas as pd

from risk_score.data import _load_raw_data


def test_load_raw_data_resolves_relative_path_from_repo_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = repo_root / "data"
    data_dir.mkdir()
    dataset_path = data_dir / "sample.tab.txt"
    dataset_path.write_text("AGE\tSEX\tBMI\tBP\tS1\tS2\tS3\tS4\tS5\tS6\tY\n1\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\n")

    fake_module_path = repo_root / "src" / "risk_score" / "data.py"
    fake_module_path.parent.mkdir(parents=True)
    fake_module_path.write_text("# test helper path anchor\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("risk_score.data.__file__", str(fake_module_path))

    df = _load_raw_data({"path": "data/sample.tab.txt"})

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6", "Y"]
