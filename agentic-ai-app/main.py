from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import build_agent, build_agent_with_parser, AsyncAgent
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage
from helpers import (
    _call_agent,
    _call_agent_with_parser,
    _result_to_text,
    summarize_conversation,
    _make_thread_id,
    extract_entities,
    SUMMARY_TURN_THRESHOLD,
)
import json
from tools import calculator, get_weather_summary, get_multiple_weather


class Query(BaseModel):
    question: str
    thread_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    print("🚀 Starting up the Async Agentic AI API...")
    agent_executer = await build_agent()
    agent_executer_with_parser = await build_agent_with_parser()
    agent_executer_for_parallel_tool_calls = AsyncAgent(
        tools=[calculator, get_weather_summary, get_multiple_weather]
    )

    app.state.agent_executer = agent_executer
    app.state.agent_executer_with_parser = agent_executer_with_parser
    app.state.agent_executer_for_parallel_tool_calls = (
        agent_executer_for_parallel_tool_calls
    )
    yield
    # Shutdown actions
    print("🛑 Shutting down the Async Agentic AI API...")
    del app.state.agent_executer


app = FastAPI(title="Async Agentic AI API", lifespan=lifespan)
# --- App-level memory stores (in-memory) ------------------------------------
# Note: for production use persistent store (Redis / Postgres). This in-memory is for demo.
if not hasattr(app.state, "conversations"):
    app.state.conversations = {}  # thread_id -> list[ (role, text) ]

if not hasattr(app.state, "entities"):
    app.state.entities = {}  # thread_id -> dict


@app.post("/ask")
async def ask_agent(query: Query):
    """Async endpoint to query the agent"""
    try:
        print(f"🤖 Processing query: {query.question}")
        agent_executer = app.state.agent_executer
        if not agent_executer:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        config = {
            "recursion_limit": 50,
            "configurable": {
                "thread_id": "main",
                "checkpoint_ns": "agent",
            },
        }

        result = await agent_executer.ainvoke(
            {
                "messages": [
                    HumanMessage(content=query.question)
                    # ("user", query.question)
                ]
            },
            config=config,
        )
        answer = result["messages"][-1].content
        return {
            "question": query.question,
            "answer": answer,
            "status": "success",
            "method": "async_agent",
        }
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# --- Main ask endpoint ------------------------------------------------------
@app.post("/ask-in-memory")
async def ask_in_memory_agent(query: Query):
    thread_id = _make_thread_id(query.thread_id)
    # initialize per-thread stores
    convo = app.state.conversations.get(thread_id, [])
    # append user message
    convo.append(("user", query.question))

    # call the agent with the conversation so far (you may tweak to pass only last N)
    try:
        # pass the convo to agent (LangGraph likes messages list)
        # if convo is empty (shouldn't), we still pass current question
        messages_to_send = convo.copy()
        print(messages_to_send)
        result = await _call_agent(
            app_state=app.state,
            thread_id=thread_id,
            messages=messages_to_send,
            recursion_limit=50,
        )

        assistant_text = _result_to_text(result).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # append assistant
    convo.append(("assistant", assistant_text))

    # persist back
    app.state.conversations[thread_id] = convo

    # --- Summarize if too long ---
    # We count number of role entries; you can choose different heuristic
    if len(convo) > SUMMARY_TURN_THRESHOLD:
        try:
            summary = await summarize_conversation(app.state, thread_id, convo)
            # Replace older messages with single summary system message.
            # Keep only last 2-4 turns after summary to preserve recency.
            tail = convo[-4:] if len(convo) >= 4 else convo
            new_convo = [("system", f"CONVERSATION_SUMMARY: {summary}")] + tail
            app.state.conversations[thread_id] = new_convo
            convo = new_convo
        except Exception as e:
            # summarization failure shouldn't block main flow
            print("Warning: summarization failed:", e)

    # --- Extract+store entities ---
    try:
        entities = await extract_entities(app.state, thread_id, convo)
        app.state.entities[thread_id] = entities
    except Exception as e:
        print("Warning: entity extraction failed:", e)

    return {
        "question": query.question,
        "answer": assistant_text,
        "status": "success",
        "thread_id": thread_id,
    }


# --- Utility endpoints ------------------------------------------------------
@app.get("/history")
async def get_history(thread_id: str | None = None):
    tid = _make_thread_id(thread_id)
    return {"thread_id": tid, "conversation": app.state.conversations.get(tid, [])}


@app.get("/entities")
async def get_entities(thread_id: str | None = None):
    tid = _make_thread_id(thread_id)
    return {"thread_id": tid, "entities": app.state.entities.get(tid, {})}


@app.post("/ask-with-parser")
async def ask_agent_with_parser(query: Query):
    """Async endpoint to query the agent with output parser"""
    thread_id = f"{_make_thread_id(query.thread_id)}_parser"
    try:
        print(f"🤖 Processing query with parser: {query.question}")

        result = await _call_agent_with_parser(
            app_state=app.state,
            thread_id=thread_id,
            question=query.question,
            recursion_limit=50,
        )
        # 🔑 Last AI message uthao
        if isinstance(result, dict) and "messages" in result:
            last_msg = result["messages"][-1]
            content = getattr(last_msg, "content", str(last_msg))
        else:
            content = str(result)

        # 🚀 Agar string hai to json.loads() se dict bana do
        try:
            parsed = json.loads(content)
        except Exception:
            # fallback if LLM ne kuch aur bhi ghusa diya
            parsed = {"raw_output": content}

        return {
            "question": query.question,
            "result": parsed,
            "status": "success",
            "method": "async_agent_with_parser",
        }
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/ask-parallel-tool-calls")
async def ask_agent_for_parallel_calls(query: Query):
    """Endpoint to query the agent which makes parallel tool calls"""
    thread_id = f"{_make_thread_id(query.thread_id)}_parallel"
    try:
        print(f"🤖 Processing query with parallel tool calls: {query.question}")

        result = await _call_agent(
            app_state=app.state,
            thread_id=thread_id,
            messages=[("user", query.question)],
            recursion_limit=50,
            agent_executor=app.state.agent_executer_for_parallel_tool_calls,
        )

        answer = _result_to_text(result).strip()
        return {
            "question": query.question,
            "answer": answer,
            "status": "success",
            "method": "async_agent_parallel_tool_calls",
        }
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
