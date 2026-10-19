"""
RescuGuard AI - Unit Tests for Predictor & Pydantic Schemas
-----------------------------------------------------------
Tests input validation boundaries, priority protocol mappings,
contributing risk factor extraction, and end-to-end model inference.
"""

import sys
import unittest
from pathlib import Path

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from pydantic import ValidationError
from app.schemas.accident import AccidentFeatureInput, PredictionResponse
from app.services.predictor import AccidentSeverityPredictor


class TestAccidentSchemas(unittest.TestCase):
    """Tests Pydantic validation rules and physical bounds."""

    def test_valid_payload_instantiation(self):
        """Ensures a realistic accident feature payload passes validation."""
        payload = AccidentFeatureInput(
            Start_Lat=9.0108,
            Start_Lng=38.7616,
            **{
                "Temperature(F)": 65.0,
                "Humidity(%)": 70.0,
                "Visibility(mi)": 8.0,
            },
            Bump=0,
            Crossing=1,
            Traffic_Signal=1,
            Weather_Condition="Clear",
            Sunrise_Sunset="Day"
        )
        self.assertEqual(payload.Start_Lat, 9.0108)
        self.assertEqual(payload.Start_Lng, 38.7616)
        self.assertEqual(payload.Sunrise_Sunset, "Day")

    def test_invalid_latitude_out_of_bounds(self):
        """Ensures latitude > 90 raises ValidationError."""
        with self.assertRaises(ValidationError):
            AccidentFeatureInput(
                Start_Lat=120.5,
                Start_Lng=38.7616,
                **{"Temperature(F)": 65.0, "Humidity(%)": 70.0, "Visibility(mi)": 8.0},
                Bump=0, Crossing=0, Traffic_Signal=0,
                Weather_Condition="Clear", Sunrise_Sunset="Day"
            )

    def test_invalid_humidity_percentage(self):
        """Ensures humidity > 100% raises ValidationError."""
        with self.assertRaises(ValidationError):
            AccidentFeatureInput(
                Start_Lat=9.0,
                Start_Lng=38.0,
                **{"Temperature(F)": 65.0, "Humidity(%)": 150.0, "Visibility(mi)": 8.0},
                Bump=0, Crossing=0, Traffic_Signal=0,
                Weather_Condition="Clear", Sunrise_Sunset="Day"
            )

    def test_invalid_solar_period(self):
        """Ensures non-standard solar period (neither Day nor Night) fails regex validation."""
        with self.assertRaises(ValidationError):
            AccidentFeatureInput(
                Start_Lat=9.0,
                Start_Lng=38.0,
                **{"Temperature(F)": 65.0, "Humidity(%)": 50.0, "Visibility(mi)": 8.0},
                Bump=0, Crossing=0, Traffic_Signal=0,
                Weather_Condition="Clear", Sunrise_Sunset="Twilight"
            )


class TestPredictorService(unittest.TestCase):
    """Tests priority mappings, risk factor extractions, and pipeline inference."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = AccidentSeverityPredictor()

    def test_priority_mapping_levels(self):
        """Verifies operational dispatch mappings for all severity tiers (1-4)."""
        mapping_l1 = self.predictor._map_priority(1)
        self.assertEqual(mapping_l1["priority"], "LOW")

        mapping_l2 = self.predictor._map_priority(2)
        self.assertEqual(mapping_l2["priority"], "MEDIUM")

        mapping_l3 = self.predictor._map_priority(3)
        self.assertEqual(mapping_l3["priority"], "HIGH")

        mapping_l4 = self.predictor._map_priority(4)
        self.assertEqual(mapping_l4["priority"], "CRITICAL")

    def test_risk_factor_extraction_night_and_fog(self):
        """Tests that adverse conditions (fog, night, freezing) generate explainable risk factors."""
        payload = AccidentFeatureInput(
            Start_Lat=9.0,
            Start_Lng=38.0,
            **{
                "Temperature(F)": 28.0,  # Freezing
                "Humidity(%)": 92.0,    # High humidity
                "Visibility(mi)": 0.5,  # Severely impaired visibility
            },
            Bump=0,
            Crossing=0,
            Traffic_Signal=0,
            Weather_Condition="Dense Fog",
            Sunrise_Sunset="Night"
        )
        factors = self.predictor._extract_contributing_risk_factors(payload)
        factor_text = " ".join(factors).lower()

        self.assertTrue(any("visibility" in f.lower() for f in factors))
        self.assertTrue(any("fog" in f.lower() for f in factors))
        self.assertTrue(any("freezing" in f.lower() or "ice" in f.lower() for f in factors))
        self.assertTrue(any("night" in f.lower() for f in factors))
        self.assertTrue(any("uncontrolled" in f.lower() for f in factors))

    def test_full_pipeline_prediction(self):
        """Tests end-to-end inference if the model pipeline artifact is loaded."""
        if not self.predictor.is_loaded:
            self.skipTest("Trained model pipeline not available in environment.")

        payload = AccidentFeatureInput(
            Start_Lat=39.8651,
            Start_Lng=-84.0587,
            **{
                "Temperature(F)": 36.9,
                "Humidity(%)": 91.0,
                "Visibility(mi)": 10.0,
            },
            Bump=0,
            Crossing=1,
            Traffic_Signal=1,
            Weather_Condition="Clear",
            Sunrise_Sunset="Day"
        )
        response = self.predictor.predict(payload)

        self.assertIsInstance(response, PredictionResponse)
        self.assertIn(response.predicted_severity, [1, 2, 3, 4])
        self.assertIn(response.priority_level, ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        self.assertGreater(len(response.dispatch_recommendation), 0)
        self.assertIsNotNone(response.inference_time_ms)

        # Probabilities should sum approximately to 1.0
        prob_sum = sum(response.probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
