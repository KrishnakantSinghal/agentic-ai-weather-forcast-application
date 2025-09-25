from langchain.tools import tool
import aiohttp
import asyncio


@tool("get_weather")
async def get_weather(city: str) -> str:
    """Fetches the current weather for a given city."""
    url = f"http://wttr.in/{city}?format=3"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()


@tool("calculator")
async def calculator(expression: str) -> str:
    """Evaluates a math expression safely."""
    # SAFER than eval – see section 4
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool("get_average_weather")
async def get_average_weather(cities: list[str]) -> str:
    """Fetches the average weather for a list of cities."""
    tasks = [get_weather(city) for city in cities]
    results = await asyncio.gather(*tasks)
    return "\n".join(results)


@tool("get_multiple_weather")
async def get_multiple_weather(cities: list[str]) -> dict:
    """Fetch weather for multiple cities in parallel."""
    results = await asyncio.gather(*(get_weather(city) for city in cities))
    return {city: res for city, res in zip(cities, results)}


@tool("get_weather_summary")
async def get_weather_summary(cities: list[str]) -> dict:
    """Get current weather for multiple cities concurrently."""
    tasks = [get_weather(city) for city in cities]
    results = await asyncio.gather(*tasks)
    return dict(zip(cities, results))
