from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Optional


class WeatherData(BaseModel):
    city: str
    country: str
    temperature: float
    condition: str
    humidity: float
    wind_speed: float


class WeatherResponse(BaseModel):
    cities: List[WeatherData]
    average_temperature: Optional[float] = None
    calculated_value: Optional[float] = None


weather_parser = PydanticOutputParser(pydantic_object=WeatherResponse)
format_instructions = weather_parser.get_format_instructions()
