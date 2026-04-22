from pydantic import BaseModel


class PatientFeatures(BaseModel):
    AGE: float
    SEX: float
    BMI: float
    BP: float
    S1: float
    S2: float
    S3: float
    S4: float
    S5: float
    S6: float


class PredictResponse(BaseModel):
    risk_score: float


class FeatureDetail(BaseModel):
    score: float
    weight: float


class ExplainResponse(BaseModel):
    risk_score: float
    feature_scores: dict[str, FeatureDetail]


class WeightsResponse(BaseModel):
    selected_features: list[str]
    weights: dict[str, float]
