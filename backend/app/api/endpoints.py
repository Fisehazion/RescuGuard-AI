from fastapi import APIRouter, HTTPException
from app.schemas.accident import AccidentFeatureInput, PredictionResponse
from app.services.predictor import AccidentSeverityPredictor

router = APIRouter()
predictor = AccidentSeverityPredictor()


@router.get("/health")
def health_check():
    return {"status": "online", "system": "Accident Severity Predictor API"}


@router.post("/predict", response_model=PredictionResponse)
def predict_accident_severity(payload: AccidentFeatureInput):
    try:
        response = predictor.predict(payload)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))