from typing import List, Tuple
import json
from parsers import format_instructions
import asyncio
from tools import get_weather_summary, calculator, get_multiple_weather


# --- Helpers -----------------------------------------------------------------
def _make_thread_id(proposed: str | None) -> str:
    return proposed if proposed else "main"


def _result_to_text(result) -> str:
    """Extract a sane assistant text from an agent result (handles multiple result shapes)."""
    if not result:
        return ""
    # common langchain-style
    if isinstance(result, dict):
        if "output" in result:
            return result["output"]
        if "output_text" in result:
            return result["output_text"]
        # LangGraph create_react_agent sometimes returns messages list
        if "messages" in result:
            msgs = result["messages"]
            # messages may be tuples or objects with .content
            for m in reversed(msgs):
                # prefer last assistant message
                if isinstance(m, tuple) and m[0] in ("assistant", "ai", "system"):
                    return m[1]
                if hasattr(m, "content"):
                    return getattr(m, "content")
        # fallback
        return str(result)
    return str(result)


async def _call_agent(
    app_state,
    thread_id: str,
    messages: List[Tuple[str, str]],
    recursion_limit: int = 25,
    agent_executor=None,
):
    """Call agent_executor. Returns raw result."""
    if not hasattr(app_state, "agent_executer") or app_state.agent_executer is None:
        raise RuntimeError("agent_executer not initialized on app_state")

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    if agent_executor is not None:
        return await agent_executor.arun({"messages": messages}, config=config)
    # agent expects {"messages": [("user", "..."), ...]} as earlier in your code
    return await app_state.agent_executer.ainvoke({"messages": messages}, config=config)


async def _call_agent_with_parser(
    app_state,
    thread_id: str,
    question: str,
    recursion_limit: int = 25,
):
    """Call agent_executor_with_parser. Returns raw result."""
    if (
        not hasattr(app_state, "agent_executer_with_parser")
        or app_state.agent_executer_with_parser is None
    ):
        raise RuntimeError("agent_executer_with_parser not initialized on app_state")

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }

    return await app_state.agent_executer_with_parser.ainvoke(
        {
            "messages": [
                ("system", "You are a weather assistant. Return only JSON."),
                ("system", format_instructions),
                ("user", question),
            ]
        },
        config=config,
    )


# --- Summary + Entity logic --------------------------------------------------
SUMMARY_TURN_THRESHOLD = (
    8  # after this many role messages (user+assistant) -> summarize
)
MAX_MESSAGES_TO_SUMMARIZE = 40  # cap on how many messages we feed into summarizer


async def summarize_conversation(
    app_state, thread_id: str, convo: List[Tuple[str, str]]
) -> str:
    """
    Ask the agent/LLM to summarize the conversation into 2-3 sentences.
    Returns the summary text.
    """
    # Build plain text of last N messages
    to_take = convo[-MAX_MESSAGES_TO_SUMMARIZE:]
    convo_text = "\n".join([f"{role.upper()}: {text}" for role, text in to_take])

    system_prompt = (
        "You are a concise summarizer. Produce a short 2-3 sentence summary "
        "of the conversation. Output only the summary text (no commentary)."
    )

    messages = [
        ("system", system_prompt),
        ("user", f"Conversation:\n\n{convo_text}\n\nPlease provide a short summary:"),
    ]

    result = await _call_agent(app_state, thread_id, messages, recursion_limit=10)
    return _result_to_text(result).strip()


async def extract_entities(
    app_state, thread_id: str, convo: List[Tuple[str, str]]
) -> dict:
    """
    Ask LLM to extract structured entities: name, location, preferences.
    Returns dict (may be empty).
    """
    to_take = convo[-MAX_MESSAGES_TO_SUMMARIZE:]
    convo_text = "\n".join([f"{role.upper()}: {text}" for role, text in to_take])

    # Ask for JSON strictly
    system_prompt = (
        "You are a JSON extractor. From the following conversation, extract three fields "
        "if present: name, location, preferences. Return a valid JSON object with keys "
        "name, location, preferences. If a field is not present, return null for it. "
        "Do not include extra commentary."
    )

    user_prompt = f"Conversation:\n\n{convo_text}\n\nReturn JSON only."

    messages = [
        ("system", system_prompt),
        ("user", user_prompt),
    ]

    result = await _call_agent(app_state, thread_id, messages, recursion_limit=10)
    text = _result_to_text(result).strip()

    # Try to parse JSON from response robustly
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        # attempt to locate a JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    # fallback: return empty structured dict
    return {"name": None, "location": None, "preferences": None}


async def generate_tool_calls(user_input: str):
    """
    LLM generates JSON with tool calls only. Do not answer directly.
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
    response = await llm.ainvoke(llm_prompt)
    import json

    try:
        return json.loads(response)
    except Exception:
        return {"tool_calls": []}


# -------------------------
# Parallel execution of tools
# -------------------------
async def execute_tools_parallel(tool_calls):
    tasks = []
    results = []

    for call in tool_calls:
        tool_name = call.get("tool")
        tool_input = call.get("tool_input")

        if tool_name == "get_weather_summary":
            tasks.append(get_weather_summary.arun(tool_input))
        elif tool_name == "calculator":
            tasks.append(calculator.arun(tool_input))
        elif tool_name == "get_multiple_weather":
            tasks.append(get_multiple_weather.arun(tool_input))
        else:
            # Unknown tool → skip
            tasks.append(asyncio.sleep(0, result=f"Unknown tool {tool_name}"))

    # Run all tools concurrently
    results = await asyncio.gather(*tasks)
    return results
