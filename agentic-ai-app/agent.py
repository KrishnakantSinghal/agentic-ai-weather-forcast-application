import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from tools import (
    calculator,
    get_weather,
    get_average_weather,
    get_multiple_weather,
    get_weather_summary,
)
from parsers import format_instructions
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
# from helpers import should_end

load_dotenv()


async def build_agent():
    llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))
    tools = [calculator, get_weather, get_average_weather]
    agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
    return agent


async def build_agent_with_parser():
    llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))
    tools = [calculator, get_weather, get_average_weather]
    weather_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a weather assistant. Always return response in JSON."),
            ("system", "{format_instructions}"),
            MessagesPlaceholder("messages"),
        ]
    ).partial(format_instructions=format_instructions)

    agent = create_react_agent(
        llm,
        tools,
        checkpointer=MemorySaver(),
        prompt=weather_prompt,
    )
    return agent


# async def build_agent_with_parallel_calls(tool_calls):
#     tasks = []
#     results = []

#     for call in tool_calls:
#         tool_name = call.get("tool")
#         tool_input = call.get("tool_input")

#         if tool_name == "get_weather_summary":
#             tasks.append(get_weather_summary.arun(tool_input))
#         elif tool_name == "calculator":
#             tasks.append(calculator.arun(tool_input))
#         elif tool_name == "get_multiple_weather":
#             tasks.append(get_multiple_weather.arun(tool_input))
#         else:
#             # Unknown tool → skip
#             tasks.append(asyncio.sleep(0, result=f"Unknown tool {tool_name}"))


#     # Run all tools concurrently
#     results = await asyncio.gather(*tasks)
#     return results
class AsyncAgent:
    def __init__(self, llm=None, tools=None):
        self.llm = llm or ChatGroq(
            model="gemma2-9b-it",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.tools = {tool.name: tool for tool in (tools or [])}

    async def generate_tool_calls(self, user_input: str):
        """
        Prompt engineering: LLM only returns tool calls in JSON
        """
        llm_prompt = f"""
You are an agent. Do NOT answer directly. Only return JSON describing the tools to call.

Format:
{{
  "tool_calls":[
    {{"tool":"get_weather", "tool_input":"City1"}},
    {{"tool":"get_weather", "tool_input":"City2"}},
    {{"tool":"calculator", "tool_input":"<City1_temp>-<City2_temp>"}}
  ]
}}

User question: {user_input}
JSON:
"""
        response = await self.llm.ainvoke(llm_prompt)
        import json

        try:
            return json.loads(response)
        except Exception:
            return {"tool_calls": []}

    async def execute_tools_parallel(self, tool_calls):
        """
        Run tools concurrently
        """
        tasks = []
        results = []

        for call in tool_calls:
            tool_name = call.get("tool")
            tool_input = call.get("tool_input")
            tool_fn = self.tools.get(tool_name)

            if tool_fn:
                tasks.append(tool_fn.arun(tool_input))
            else:
                tasks.append(asyncio.sleep(0, result=f"Unknown tool {tool_name}"))

        results = await asyncio.gather(*tasks)
        return results

    async def arun(self, query: str):
        """
        Main agent run method
        """
        tool_calls_json = await self.generate_tool_calls(query)
        tool_calls = tool_calls_json.get("tool_calls", [])
        if not tool_calls:
            return "No tools identified for this query."

        tool_results = await self.execute_tools_parallel(tool_calls)

        # Format final answer
        final_answer = []
        for call, res in zip(tool_calls, tool_results):
            final_answer.append(f"{call['tool']}({call['tool_input']}) => {res}")

        return " ; ".join(final_answer)
