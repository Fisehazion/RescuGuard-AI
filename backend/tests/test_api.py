"""
RescuGuard AI - Integration Tests for API Endpoints
----------------------------------------------------
Tests /api/v1/health and /api/v1/predict endpoint behavior and responses.
"""

import sys
import unittest
from pathlib import Path

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.api.endpoints import health_check, predict_accident_severity
from app.schemas.accident import AccidentFeatureInput, HealthResponse, PredictionResponse


class TestAPIEndpoints(unittest.TestCase):
    """Direct integration tests for API endpoint handlers."""

    def test_health_check_endpoint(self):
        """Verifies /health endpoint returns online status and service metadata."""
        response = health_check()
        self.assertIsInstance(response, HealthResponse)
        self.assertEqual(response.status, "online")
        self.assertIn("RescuGuard", response.service)
        self.assertIsNotNone(response.model_loaded)

    def test_predict_endpoint_execution(self):
        """Verifies /predict endpoint processes valid AccidentFeatureInput payloads."""
        payload = AccidentFeatureInput(
            Start_Lat=9.0108,
            Start_Lng=38.7616,
            **{
                "Temperature(F)": 68.0,
                "Humidity(%)": 60.0,
                "Visibility(mi)": 10.0,
            },
            Bump=0,
            Crossing=1,
            Traffic_Signal=1,
            Weather_Condition="Clear",
            Sunrise_Sunset="Day"
        )
        response = predict_accident_severity(payload)
        self.assertIsInstance(response, PredictionResponse)
        self.assertIn(response.predicted_severity, [1, 2, 3, 4])
        self.assertIsInstance(response.dispatch_recommendation, list)
        self.assertIsInstance(response.contributing_risk_factors, list)


if __name__ == '__main__':
    unittest.main()
