from pydantic import BaseModel, Field


class WeatherResponse(BaseModel):
    city: str = Field(..., description="City name")
    temperature: float = Field(..., description="Temperature in Celsius")
    condition: str = Field(..., description="Weather condition like sunny, rainy")
