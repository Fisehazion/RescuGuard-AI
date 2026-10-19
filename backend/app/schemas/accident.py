"""
RescuGuard AI - Pydantic Data Schemas
-------------------------------------
Defines strict input validation schemas with domain physical bounds,
response structures, and explainability payloads for the emergency API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class AccidentFeatureInput(BaseModel):
    """
    Input schema representing incident conditions collected by emergency call takers,
    traffic monitoring sensors, or connected vehicle telemetry.
    Includes physical validation bounds.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Start_Lat": 9.0108,
                "Start_Lng": 38.7616,
                "Temperature(F)": 62.5,
                "Humidity(%)": 78.0,
                "Visibility(mi)": 4.5,
                "Bump": 0,
                "Crossing": 1,
                "Traffic_Signal": 1,
                "Weather_Condition": "Rain",
                "Sunrise_Sunset": "Night"
            }
        }
    )

    Start_Lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="GPS Latitude in decimal degrees (-90 to 90)"
    )
    Start_Lng: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="GPS Longitude in decimal degrees (-180 to 180)"
    )
    Temperature_F: float = Field(
        ...,
        alias="Temperature(F)",
        ge=-80.0,
        le=150.0,
        description="Ambient temperature in Fahrenheit (-80 to 150 °F)"
    )
    Humidity_Percent: float = Field(
        ...,
        alias="Humidity(%)",
        ge=0.0,
        le=100.0,
        description="Relative atmospheric humidity percentage (0 to 100%)"
    )
    Visibility_Mi: float = Field(
        ...,
        alias="Visibility(mi)",
        ge=0.0,
        le=50.0,
        description="Atmospheric visibility distance in miles (0 to 50 mi)"
    )
    Bump: int = Field(
        0,
        ge=0,
        le=1,
        description="Binary flag: 1 if speed bump is present at site, else 0"
    )
    Crossing: int = Field(
        0,
        ge=0,
        le=1,
        description="Binary flag: 1 if pedestrian crossing is present, else 0"
    )
    Traffic_Signal: int = Field(
        0,
        ge=0,
        le=1,
        description="Binary flag: 1 if traffic control signal is present, else 0"
    )
    Weather_Condition: str = Field(
        "Clear",
        min_length=1,
        max_length=60,
        description="Reported meteorological condition (e.g., Clear, Rain, Fog, Snow, Overcast)"
    )
    Sunrise_Sunset: str = Field(
        "Day",
        pattern=r"^(Day|Night)$",
        description="Solar period at time of incident: 'Day' or 'Night'"
    )


class PredictionResponse(BaseModel):
    """
    Output schema returned to emergency dispatch operators containing predicted severity,
    dispatch priority, operational protocols, class probabilities, and explainable risk factors.
    """
    predicted_severity: int = Field(
        ...,
        ge=1,
        le=4,
        description="Predicted accident severity category (Scale: 1 = Minor to 4 = Critical)"
    )
    severity_label: str = Field(
        ...,
        description="Human-readable severity classification label"
    )
    priority_level: str = Field(
        ...,
        description="Standardized dispatch priority code (LOW, MEDIUM, HIGH, CRITICAL)"
    )
    response_protocol: str = Field(
        ...,
        description="Operational tactical protocol title prescribed for command dispatch"
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Calibrated probability distribution across all four severity tiers"
    )
    dispatch_recommendation: List[str] = Field(
        ...,
        description="List of concrete tactical emergency resource dispatch instructions"
    )
    contributing_risk_factors: List[str] = Field(
        default_factory=list,
        description="Key environmental and road infrastructure risk factors detected"
    )
    inference_time_ms: Optional[float] = Field(
        None,
        description="Model execution latency in milliseconds"
    )


class HealthResponse(BaseModel):
    """
    Schema for system health check and model availability monitoring.
    """
    status: str
    service: str
    version: str
    model_loaded: bool
    model_path: str