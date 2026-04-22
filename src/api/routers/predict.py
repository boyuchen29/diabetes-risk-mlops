from fastapi import APIRouter, Request
from ..schemas import PatientFeatures, PredictResponse, ExplainResponse, WeightsResponse

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
def predict(patient: PatientFeatures, request: Request):
    risk_score = request.app.state.scorer.score(patient.model_dump())
    return PredictResponse(risk_score=risk_score)


@router.post("/explain", response_model=ExplainResponse)
def predict_explain(patient: PatientFeatures, request: Request):
    result = request.app.state.scorer.explain(patient.model_dump())
    return ExplainResponse(**result)


@router.get("/weights", response_model=WeightsResponse)
def predict_weights(request: Request):
    weights = request.app.state.scorer.weights
    return WeightsResponse(
        selected_features=list(weights.keys()),
        weights=weights,
    )
