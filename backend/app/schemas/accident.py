from pydantic import BaseModel, Field
from typing import List, Dict


class AccidentFeatureInput(BaseModel):
    Start_Lat: float = Field(..., example=39.8651, description="Start Latitude coordinate")
    Start_Lng: float = Field(..., example=-84.0587, description="Start Longitude coordinate")
    Temperature_F: float = Field(..., alias="Temperature(F)", example=36.9, description="Temperature in Fahrenheit")
    Humidity_Percent: float = Field(..., alias="Humidity(%)", example=91.0, description="Humidity percentage")
    Visibility_Mi: float = Field(..., alias="Visibility(mi)", example=10.0, description="Visibility in miles")
    Bump: int = Field(0, ge=0, le=1, description="1 if speed bump present, else 0")
    Crossing: int = Field(0, ge=0, le=1, description="1 if pedestrian crossing present, else 0")
    Traffic_Signal: int = Field(0, ge=0, le=1, description="1 if traffic signal present, else 0")
    Weather_Condition: str = Field("Clear", description="Weather condition e.g. Clear, Fair, Rain, Fog, Snow")
    Sunrise_Sunset: str = Field("Day", description="Day or Night")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    predicted_severity: int
    severity_label: str
    priority_level: str
    response_protocol: str
    probabilities: Dict[str, float]
    dispatch_recommendation: List[str]