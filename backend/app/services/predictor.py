import joblib
import pandas as pd
from typing import Dict, Any
from app.config import settings
from app.schemas.accident import AccidentFeatureInput, PredictionResponse


class AccidentSeverityPredictor:
    """Singleton service for predicting incident severity based on trained model pipelines."""

    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AccidentSeverityPredictor, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self) -> None:
        if not settings.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file missing at '{settings.MODEL_PATH}'. "
                "Execute 'python train.py' inside backend/ first."
            )
        self._pipeline = joblib.load(settings.MODEL_PATH)

    def _map_priority(self, severity: int) -> Dict[str, Any]:
        mapping = {
            1: {
                "label": "Minor Incident (Level 1)",
                "priority": "LOW",
                "protocol": "Routine Traffic Operations Control",
                "recommendation": ["Dispatch Single Traffic Patrol Unit", "Log Spatial Coordinates"]
            },
            2: {
                "label": "Moderate Incident (Level 2)",
                "priority": "MEDIUM",
                "protocol": "Traffic Diversion & Secondary Support",
                "recommendation": ["Dispatch Patrol Unit", "Standby Regional Medical Team"]
            },
            3: {
                "label": "Severe Accident (Level 3)",
                "priority": "HIGH",
                "protocol": "Urgent Medical & Rescue Operations",
                "recommendation": ["Dispatch 2 Emergency Ambulances", "Heavy Rescue Squad", "Close Corridor Lane"]
            },
            4: {
                "label": "Critical Highway Emergency (Level 4)",
                "priority": "CRITICAL",
                "protocol": "Mass Casualty & Major Arterial Emergency",
                "recommendation": ["Immediate Trauma Dispatch", "Fire & Extrication Response", "Full Sector Traffic Reroute"]
            }
        }
        return mapping.get(severity, mapping[2])

    def predict(self, input_data: AccidentFeatureInput) -> PredictionResponse:
        # Reconstruct exact column names for the trained scikit-learn pipeline
        raw_dict = input_data.model_dump(by_alias=True)
        
        # Ensure DataFrame columns match exact order expected by preprocessor
        feature_order = [
            'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
            'Bump', 'Crossing', 'Traffic_Signal', 'Weather_Condition', 'Sunrise_Sunset'
        ]
        
        df = pd.DataFrame([raw_dict])[feature_order]

        pred_class = int(self._pipeline.predict(df)[0])
        probs = self._pipeline.predict_proba(df)[0]
        classes = self._pipeline.classes_

        prob_dict = {f"Severity {cls}": round(float(prob), 4) for cls, prob in zip(classes, probs)}
        meta = self._map_priority(pred_class)

        return PredictionResponse(
            predicted_severity=pred_class,
            severity_label=meta["label"],
            priority_level=meta["priority"],
            response_protocol=meta["protocol"],
            probabilities=prob_dict,
            dispatch_recommendation=meta["recommendation"]
        )