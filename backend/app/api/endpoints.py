"""
RescuGuard AI - REST API Endpoints Router
------------------------------------------
Defines operational API routes for real-time accident severity prediction,
model health telemetry, and emergency dispatch decision support.
"""

from fastapi import APIRouter, HTTPException, status
from app.config import settings
from app.schemas.accident import AccidentFeatureInput, PredictionResponse, HealthResponse
from app.services.predictor import AccidentSeverityPredictor

router = APIRouter()
predictor = AccidentSeverityPredictor()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health & Model Readiness Probe",
    description="Returns the operational status of the API service and the machine learning pipeline."
)
def health_check() -> HealthResponse:
    """
    Health check endpoint for Docker container orchestration and load balancer probes.
    """
    return HealthResponse(
        status="online",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        model_loaded=predictor.is_loaded,
        model_path=str(settings.MODEL_PATH)
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Accident Severity & Generate Dispatch Strategy",
    description=(
        "Analyzes atmospheric, roadway, and spatial telemetry to predict accident severity "
        "and prescribes automated emergency triage protocols."
    )
)
def predict_accident_severity(payload: AccidentFeatureInput) -> PredictionResponse:
    """
    Processes incident parameters through the trained ML pipeline and outputs triage guidance.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Prediction engine is unavailable because the model artifact is not yet loaded. "
                "Ensure 'model_pipeline.joblib' is present in 'data/processed/' or run 'python train.py'."
            )
        )

    try:
        response = predictor.predict(payload)
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure encountered: {str(exc)}"
        )