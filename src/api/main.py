import os
from contextlib import asynccontextmanager
import mlflow
from fastapi import FastAPI
from .scorer import Scorer
from .middleware.auth import ApiKeyMiddleware
from .routers.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri("databricks")
    app.state.scorer = Scorer.from_mlflow(os.environ["MLFLOW_RUN_ID"])
    yield


app = FastAPI(title="diabetes-risk-api", version="0.1.0", lifespan=lifespan)
app.add_middleware(ApiKeyMiddleware)
app.include_router(predict_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"name": "diabetes-risk-api", "version": "0.1.0"}
