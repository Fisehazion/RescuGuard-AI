"""
RescuGuard AI - Model Prediction & Dispatch Triage Service
-----------------------------------------------------------
Loads the scikit-learn machine learning pipeline artifact and transforms raw
incident parameters into multi-class severity predictions, probability distributions,
contributing risk factors, and actionable dispatch protocols.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib
import pandas as pd

from app.config import settings
from app.schemas.accident import AccidentFeatureInput, PredictionResponse

logger = logging.getLogger("rescu_guard.predictor")


class AccidentSeverityPredictor:
    """
    Singleton service managing the lifecycle of the trained Random Forest pipeline
    and mapping statistical outputs into operational C4ISR emergency triage protocols.
    """

    _instance: Optional["AccidentSeverityPredictor"] = None
    _pipeline = None
    _is_loaded: bool = False

    def __new__(cls) -> "AccidentSeverityPredictor":
        if cls._instance is None:
            cls._instance = super(AccidentSeverityPredictor, cls).__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self) -> None:
        """Initializes model loading with graceful degradation if artifact is missing."""
        self._load_model()

    def _load_model(self) -> None:
        """Attempts to load the serialized pipeline from the configured path."""
        model_path: Path = settings.MODEL_PATH
        if not model_path.exists():
            logger.warning(
                f"[WARN] Model artifact not found at '{model_path}'. "
                "Predictor will operate in uninitialized mode until train.py is executed."
            )
            self._pipeline = None
            self._is_loaded = False
            return

        try:
            logger.info(f"[INFO] Loading ML pipeline artifact from '{model_path}'...")
            self._pipeline = joblib.load(model_path)
            self._is_loaded = True
            logger.info("[SUCCESS] Pipeline successfully loaded and primed for inference.")
        except Exception as exc:
            logger.error(f"[ERROR] Failed to load model artifact: {exc}")
            self._pipeline = None
            self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Returns True if the underlying scikit-learn pipeline is loaded and ready."""
        return self._is_loaded and self._pipeline is not None

    def _extract_contributing_risk_factors(self, data: AccidentFeatureInput) -> List[str]:
        """
        Derives operational domain risk factors contributing to the incident severity.
        Provides dispatchers with immediate contextual explainability.
        """
        factors: List[str] = []

        # Atmospheric visibility
        if data.Visibility_Mi < 2.0:
            factors.append(f"Severely impaired visibility ({data.Visibility_Mi} miles)")
        elif data.Visibility_Mi < 5.0:
            factors.append(f"Moderate visibility constraint ({data.Visibility_Mi} miles)")

        # Adverse meteorological events
        adverse_conditions = ["rain", "snow", "fog", "haze", "ice", "thunderstorm", "sleet", "storm"]
        weather_lower = data.Weather_Condition.lower()
        if any(term in weather_lower for term in adverse_conditions):
            factors.append(f"Hazardous atmospheric event: {data.Weather_Condition}")

        # Road surface temperature & freezing risk
        if data.Temperature_F <= 32.0:
            factors.append(f"Freezing temperature ({data.Temperature_F}°F) — black ice / skidding hazard")
        elif data.Temperature_F >= 95.0:
            factors.append(f"Extreme heat ({data.Temperature_F}°F) — tire blowout / engine strain risk")

        # Atmospheric humidity
        if data.Humidity_Percent >= 85.0:
            factors.append(f"High relative humidity ({data.Humidity_Percent}%) — wet road friction loss")

        # Temporal hazard
        if data.Sunrise_Sunset == "Night":
            factors.append("Nighttime operational window — reduced peripheral optical illumination")

        # Infrastructure risk factors
        if data.Traffic_Signal == 0 and data.Crossing == 0:
            factors.append("Uncontrolled roadway sector (no traffic signal or crossing protection)")
        if data.Bump == 1:
            factors.append("Vertical road deflection / speed bump present")

        if not factors:
            factors.append("Standard ambient conditions; elevated incident probability driven by spatial corridor metrics")

        return factors

    def _map_priority(self, severity: int) -> Dict[str, Any]:
        """
        Maps statistical severity levels (1 to 4) to concrete C4ISR command protocols
        and automated emergency asset dispatch recommendations.
        """
        protocols = {
            1: {
                "label": "Minor Incident (Severity Level 1)",
                "priority": "LOW",
                "protocol": "Standard Traffic Operations & Route Monitoring",
                "recommendation": [
                    "Dispatch single municipal traffic patrol unit",
                    "Log spatial GPS coordinates and clear shoulder lane",
                    "Monitor corridor CCTV feeds for traffic backlog"
                ]
            },
            2: {
                "label": "Moderate Incident (Severity Level 2)",
                "priority": "MEDIUM",
                "protocol": "Active Traffic Diversion & Secondary Support",
                "recommendation": [
                    "Dispatch rapid traffic control patrol unit",
                    "Place regional paramedic crew on standby",
                    "Deploy variable message signs (VMS) to reduce approach speeds",
                    "Prepare towing and roadway clearance assets"
                ]
            },
            3: {
                "label": "Severe Accident (Severity Level 3)",
                "priority": "HIGH",
                "protocol": "Urgent Trauma Care & Arterial Corridor Evacuation",
                "recommendation": [
                    "Immediate dispatch of 2 Advanced Life Support (ALS) ambulances",
                    "Deploy heavy rescue squad with hydraulic extrication tools",
                    "Enforce emergency perimeter cordon and close inbound corridor lane",
                    "Alert nearest regional trauma center emergency ward"
                ]
            },
            4: {
                "label": "Critical Highway Emergency (Severity Level 4)",
                "priority": "CRITICAL",
                "protocol": "Mass Casualty & Major Multi-Agency Incident Command",
                "recommendation": [
                    "Activate Level-4 Mass Casualty Incident (MCI) Protocol",
                    "Immediate priority dispatch of multi-unit trauma & aeromedical triage",
                    "Full sector traffic shutdown and initiate arterial reroute protocols",
                    "Deploy specialized hazardous materials and fire-extrication task force",
                    "Establish incident command post for unified inter-agency coordination"
                ]
            }
        }
        return protocols.get(severity, protocols[2])

    def predict(self, input_data: AccidentFeatureInput) -> PredictionResponse:
        """
        Executes model inference on validated input parameters and returns the full triage response.
        """
        if not self.is_loaded:
            # Re-attempt loading in case training finished after server start
            self._load_model()
            if not self.is_loaded:
                raise FileNotFoundError(
                    f"Trained model artifact missing at '{settings.MODEL_PATH}'. "
                    "Please run 'python train.py' inside the backend directory to train the model."
                )

        start_time = time.perf_counter()

        # Reconstruct feature dictionary formatted for trained pipeline
        raw_dict = input_data.model_dump(by_alias=True)

        feature_order = [
            'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
            'Bump', 'Crossing', 'Traffic_Signal', 'Weather_Condition', 'Sunrise_Sunset'
        ]

        # Construct single-row DataFrame preserving explicit feature ordering
        df = pd.DataFrame([raw_dict])[feature_order]

        # Model Inference
        pred_class = int(self._pipeline.predict(df)[0])
        probabilities = self._pipeline.predict_proba(df)[0]
        classes = self._pipeline.classes_

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Build calibrated probability dictionary
        prob_dict = {
            f"Severity {cls}": round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }

        # Operational protocol mapping & explainability
        meta = self._map_priority(pred_class)
        risk_factors = self._extract_contributing_risk_factors(input_data)

        return PredictionResponse(
            predicted_severity=pred_class,
            severity_label=meta["label"],
            priority_level=meta["priority"],
            response_protocol=meta["protocol"],
            probabilities=prob_dict,
            dispatch_recommendation=meta["recommendation"],
            contributing_risk_factors=risk_factors,
            inference_time_ms=inference_time_ms
        )