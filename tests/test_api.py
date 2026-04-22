import pytest
from fastapi.testclient import TestClient
from api.main import app
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
def client(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "fake-run-id")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setattr(Scorer, "from_mlflow", staticmethod(lambda run_id: Scorer(SCORES, WEIGHTS, SCHEMA)))
    with TestClient(app) as c:
        yield c


def test_predict_returns_200(client):
    resp = client.post("/predict", json=PATIENT, headers={"Authorization": "Bearer test-key"})
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["risk_score"] <= 100.0


def test_predict_missing_auth_returns_401(client):
    resp = client.post("/predict", json=PATIENT)
    assert resp.status_code == 401


def test_predict_wrong_key_returns_401(client):
    resp = client.post("/predict", json=PATIENT, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_health_no_auth_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_missing_feature_returns_422(client):
    incomplete = {k: v for k, v in PATIENT.items() if k != "BMI"}
    resp = client.post("/predict", json=incomplete, headers={"Authorization": "Bearer test-key"})
    assert resp.status_code == 422


def test_explain_includes_all_features(client):
    resp = client.post("/predict/explain", json=PATIENT, headers={"Authorization": "Bearer test-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["feature_scores"].keys()) == set(WEIGHTS.keys())


def test_weights_endpoint_returns_selected_features(client):
    resp = client.get("/predict/weights", headers={"Authorization": "Bearer test-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["selected_features"]) == set(WEIGHTS.keys())
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-6


def test_root_no_auth_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "diabetes-risk-api"


def test_predict_unset_api_key_returns_401(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "fake-run-id")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(Scorer, "from_mlflow", staticmethod(lambda run_id: Scorer(SCORES, WEIGHTS, SCHEMA)))
    with TestClient(app) as c:
        resp = c.get("/predict/weights", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401
