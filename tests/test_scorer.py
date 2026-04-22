import json
import pytest
from api.scorer import Scorer

SCORES = {
    "bp":  {"low": 0.0, "normal": 500.0, "elevated": 750.0, "high": 1000.0},
    "bmi": {"underweight": 0.0, "normal": 200.0, "overweight": 600.0, "obese": 1000.0},
    "s4":  {"optimal": 0.0, "normal": 500.0, "high": 1000.0},
    "s5":  {"low": 0.0, "normal": 400.0, "high": 1000.0},
}
WEIGHTS = {"bp": 0.28, "bmi": 0.35, "s4": 0.18, "s5": 0.19}
SCHEMA = [
    {"feature": "bp",  "column": "BP",  "type": "cut",
     "bins": [-1e308, 80, 90, 120, 1e308], "labels": ["low", "normal", "elevated", "high"]},
    {"feature": "bmi", "column": "BMI", "type": "cut",
     "bins": [-1e308, 18.5, 25, 30, 1e308], "labels": ["underweight", "normal", "overweight", "obese"]},
    {"feature": "s4",  "column": "S4",  "type": "cut",
     "bins": [-1e308, 3.5, 5, 1e308], "labels": ["optimal", "normal", "high"]},
    {"feature": "s5",  "column": "S5",  "type": "cut",
     "bins": [-1e308, 4.3, 4.7, 1e308], "labels": ["low", "normal", "high"]},
]

PATIENT = {"AGE": 45, "SEX": 2, "BMI": 32.1, "BP": 95.0,
           "S1": 210.0, "S2": 130.0, "S3": 45.0, "S4": 4.2, "S5": 4.6, "S6": 92.0}


@pytest.fixture
def scorer():
    return Scorer(SCORES, WEIGHTS, SCHEMA)


def test_risk_score_in_range(scorer):
    result = scorer.score(PATIENT)
    assert 0.0 <= result <= 100.0


def test_explain_includes_all_selected_features(scorer):
    result = scorer.explain(PATIENT)
    assert set(result["feature_scores"].keys()) == set(WEIGHTS.keys())
    for detail in result["feature_scores"].values():
        assert "score" in detail
        assert "weight" in detail


def test_weights_sum_to_one(scorer):
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6


def test_init_raises_on_missing_scores_entry():
    bad_scores = {k: v for k, v in SCORES.items() if k != "bmi"}
    with pytest.raises(ValueError, match="missing from scores"):
        Scorer(bad_scores, WEIGHTS, SCHEMA)


def test_init_raises_on_missing_schema_entry():
    bad_schema = [s for s in SCHEMA if s["feature"] != "bmi"]
    with pytest.raises(ValueError, match="missing from schema"):
        Scorer(SCORES, WEIGHTS, bad_schema)


def test_from_mlflow_loads_artifacts(tmp_path, monkeypatch):
    (tmp_path / "scores.json").write_text(json.dumps(SCORES))
    (tmp_path / "weights.json").write_text(json.dumps(WEIGHTS))
    (tmp_path / "categorization_schema.json").write_text(json.dumps(SCHEMA))
    monkeypatch.setattr(
        "mlflow.artifacts.download_artifacts",
        lambda run_id, artifact_path: str(tmp_path),
    )
    scorer = Scorer.from_mlflow("fake-run-id")
    assert scorer.scores == SCORES
    assert scorer.weights == WEIGHTS
    assert scorer.schema == SCHEMA
    assert 0.0 <= scorer.score(PATIENT) <= 100.0
