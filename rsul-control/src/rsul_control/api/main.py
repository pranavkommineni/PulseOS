from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Any

from rsul_control.prediction.engine import predict_payload

app = FastAPI(title="PulseOS RSUL Control", version="1.0.0")

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: Any

@app.get("/health")
def health():
    return {"status": "ok", "service": "rsul-control"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        payload = request.model_dump()
        return predict_payload(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
