"""
Module 07: Complete Agent
==========================
A complete, production-ready AI assistant combining all concepts:
- Tools with ReAct loop
- Persistent memory
- Streaming responses
- Input validation
- Rate limiting
- Structured logging
"""

import asyncio
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict):
    """State for the complete agent."""
    messages: Annotated[list[BaseMessage], add_messages]


# =============================================================================
# Tool Definitions
# =============================================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A valid math expression (e.g., "2 + 2", "10 * 5")
    
    Returns:
        The result as a string
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Only basic math operations allowed"
    
    if len(expression) > 100:
        return "Error: Expression too long"
    
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for information.
    
    Args:
        query: The search query
    
    Returns:
        Relevant information
    """
    knowledge_base = {
        "langgraph": (
            "LangGraph is a library for building stateful, multi-actor applications "
            "with LLMs. It uses a graph-based approach where nodes are functions and "
            "edges define the flow between them."
        ),
        "tools": (
            "Tools in LangGraph allow LLMs to take actions. They are defined using "
            "the @tool decorator and executed via ToolNode."
        ),
        "memory": (
            "LangGraph supports memory through checkpointers. MemorySaver is for "
            "development, SqliteSaver for local persistence, and PostgresSaver "
            "for production deployments."
        ),
        "streaming": (
            "Streaming in LangGraph uses astream_events() to deliver tokens "
            "in real-time as the LLM generates them."
        ),
        "react": (
            "ReAct (Reasoning + Acting) is a pattern where the agent alternates "
            "between reasoning (thinking) and acting (using tools) until it can "
            "provide a final answer."
        ),
        "agent": (
            "An AI agent is a system that can reason about tasks, make decisions, "
            "use tools to take actions, and learn from feedback. Modern agents "
            "like Claude, Gemini, and ChatGPT are examples."
        ),
    }
    
    query_lower = query.lower()
    results = []
    
    for key, value in knowledge_base.items():
        if key in query_lower:
            results.append(value)
    
    if results:
        return " ".join(results)
    
    return f"No specific information found for: {query}"


@tool
def string_operations(operation: str, text: str) -> str:
    """Perform string operations.
    
    Args:
        operation: The operation (length, upper, lower, reverse)
        text: The text to process
    
    Returns:
        Result of the operation
    """
    operation = operation.lower().strip()
    
    if operation == "length":
        return f"The text has {len(text)} characters"
    elif operation == "upper":
        return f"Uppercase: {text.upper()}"
    elif operation == "lower":
        return f"Lowercase: {text.lower()}"
    elif operation == "reverse":
        return f"Reversed: {text[::-1]}"
    else:
        return f"Unknown operation: {operation}. Use: length, upper, lower, or reverse"


# Collect all tools
tools = [calculator, get_current_time, search_knowledge, string_operations]


# =============================================================================
# LLM and Agent Configuration
# =============================================================================

def get_llm():
    """Create and return the LLM instance."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )


llm = get_llm()
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# Graph Nodes
# =============================================================================

async def agent_node(state: AgentState) -> dict:
    """Main agent reasoning node."""
    messages = state["messages"]
    logger.info(f"Agent processing {len(messages)} messages")
    
    response = await llm_with_tools.ainvoke(messages)
    
    return {"messages": [response]}


tool_node = ToolNode(tools)


# =============================================================================
# Graph Construction
# =============================================================================

def build_agent(checkpointer=None):
    """Build the complete agent graph."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    
    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")
    
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# Utilities
# =============================================================================

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 20, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = timedelta(seconds=window_seconds)
        self.calls: dict[str, list[datetime]] = defaultdict(list)
    
    def check(self, user_id: str) -> tuple[bool, int]:
        """Check if user can make a call. Returns (allowed, remaining)."""
        now = datetime.now()
        cutoff = now - self.window
        
        self.calls[user_id] = [t for t in self.calls[user_id] if t > cutoff]
        
        remaining = max(0, self.max_calls - len(self.calls[user_id]))
        
        if remaining > 0:
            self.calls[user_id].append(now)
            return True, remaining - 1
        
        return False, 0


def validate_input(text: str) -> tuple[bool, str]:
    """Validate user input."""
    if not text or not text.strip():
        return False, "Input cannot be empty"
    
    if len(text) > 4000:
        return False, "Input too long (max 4000 characters)"
    
    # Check for injection patterns
    patterns = [
        r"ignore (?:all )?previous",
        r"disregard (?:all )?prior",
        r"new system prompt",
    ]
    
    for pattern in patterns:
        if re.search(pattern, text.lower()):
            return False, "Invalid input detected"
    
    return True, text.strip()


# =============================================================================
# Streaming
# =============================================================================

async def stream_response(agent, messages: list, config: dict):
    """Stream agent response with tool visibility."""
    collected_response = []
    
    async for event in agent.astream_events(
        {"messages": messages},
        config=config,
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                collected_response.append(chunk.content)
                print(chunk.content, end="", flush=True)
                
        elif kind == "on_tool_start":
            tool_name = event["name"]
            print(f"\n[Tool: {tool_name}]", flush=True)
            
        elif kind == "on_tool_end":
            output = str(event["data"].get("output", ""))[:80]
            print(f"[Result: {output}]", flush=True)
    
    print()
    return "".join(collected_response)


# =============================================================================
# Interactive Session
# =============================================================================

async def run_interactive():
    """Run an interactive chat session."""
    print("=" * 70)
    print("Complete AI Assistant")
    print("=" * 70)
    print("Features: Tools, Memory, Streaming, Rate Limiting")
    print("Available tools: calculator, get_current_time, search_knowledge, string_operations")
    print("Commands: 'quit' to exit, 'new' for new session, 'history' to view")
    print("=" * 70)
    
    # Create agent with memory
    memory = MemorySaver()
    agent = build_agent(checkpointer=memory)
    
    # Initialize
    thread_id = f"session-{datetime.now().timestamp()}"
    config = {"configurable": {"thread_id": thread_id}}
    rate_limiter = RateLimiter()
    user_id = "default"
    
    # System message
    system_msg = SystemMessage(
        content=(
            "You are a helpful AI assistant with access to tools. "
            "Use tools when appropriate to provide accurate answers. "
            "Remember what the user tells you across the conversation. "
            "Be concise but thorough."
        )
    )
    
    await agent.ainvoke({"messages": [system_msg]}, config=config)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        # Commands
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
            
        if user_input.lower() == "new":
            thread_id = f"session-{datetime.now().timestamp()}"
            config = {"configurable": {"thread_id": thread_id}}
            await agent.ainvoke({"messages": [system_msg]}, config=config)
            print(f"[New session started: {thread_id[:20]}...]")
            continue
            
        if user_input.lower() == "history":
            state = agent.get_state(config)
            msgs = state.values.get("messages", [])
            print(f"\n[Conversation history: {len(msgs)} messages]")
            for i, msg in enumerate(msgs[-10:]):  # Show last 10
                role = type(msg).__name__.replace("Message", "")
                content = str(msg.content)[:60]
                print(f"  {i+1}. [{role}] {content}...")
            continue
        
        # Validate input
        valid, result = validate_input(user_input)
        if not valid:
            print(f"[Error] {result}")
            continue
        
        # Check rate limit
        allowed, remaining = rate_limiter.check(user_id)
        if not allowed:
            print("[Rate limited] Please wait before sending more messages.")
            continue
        
        # Stream response
        print(f"[Calls remaining: {remaining}]")
        print("Assistant: ", end="", flush=True)
        
        await stream_response(
            agent,
            [HumanMessage(content=user_input)],
            config
        )


# =============================================================================
# Sync API (for imports)
# =============================================================================

def create_agent():
    """Create agent instance for external use."""
    memory = MemorySaver()
    return build_agent(checkpointer=memory)


def invoke_agent(agent, message: str, thread_id: str) -> str:
    """Synchronous invoke for simple use cases."""
    import asyncio
    
    config = {"configurable": {"thread_id": thread_id}}
    
    async def _invoke():
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        return result["messages"][-1].content
    
    return asyncio.run(_invoke())


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the complete agent."""
    asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
