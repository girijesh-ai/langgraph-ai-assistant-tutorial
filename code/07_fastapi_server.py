"""
Module 07: FastAPI Server
==========================
A production API server for the AI assistant.

Run with: uvicorn 07_fastapi_server:app --reload
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict

# Load environment
load_dotenv()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="AI Assistant API",
    description="LangGraph-powered AI assistant with tools and memory",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# State and Tools
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Only basic math operations allowed"
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base."""
    kb = {
        "langgraph": "LangGraph is a library for building stateful LLM applications.",
        "tools": "Tools allow LLMs to take actions.",
        "memory": "Memory enables persistent conversations.",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return f"No information found for: {query}"


tools = [calculator, get_current_time, search_knowledge]


# =============================================================================
# Agent Setup
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
llm_with_tools = llm.bind_tools(tools)


async def agent_node(state):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)
memory = MemorySaver()

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
graph.add_edge("tools", "agent")

agent = graph.compile(checkpointer=memory)


# =============================================================================
# API Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoints."""
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = Field(default_factory=lambda: f"api-{datetime.now().timestamp()}")


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""
    response: str
    thread_id: str
    tool_calls: list[str] = []


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response."""
    config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        # Extract tool calls
        tool_calls = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc["name"])
        
        return ChatResponse(
            response=result["messages"][-1].content,
            thread_id=request.thread_id,
            tool_calls=tool_calls
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream response tokens."""
    config = {"configurable": {"thread_id": request.thread_id}}
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield chunk.content
                        
        except Exception as e:
            yield f"\n[Error: {str(e)}]"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.get("/threads/{thread_id}/history")
async def get_history(thread_id: str):
    """Get conversation history for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = agent.get_state(config)
        messages = state.values.get("messages", [])
        
        history = []
        for msg in messages:
            history.append({
                "role": type(msg).__name__.replace("Message", "").lower(),
                "content": str(msg.content)[:500]
            })
        
        return {
            "thread_id": thread_id,
            "message_count": len(history),
            "messages": history[-20:]  # Return last 20 messages
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
