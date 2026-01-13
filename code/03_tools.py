"""
Module 03: Tools and Tool Calling
==================================
A LangGraph agent with tool calling capabilities:
- Custom tools with @tool decorator
- ToolNode for automatic execution
- Conditional routing with tools_condition
- ReAct-style reasoning loop
"""

import asyncio
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()


# -----------------------------------------------------------------------------
# State Definition
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    """State for the tool-calling agent."""
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------------------------------------------------------
# Tool Definitions
# -----------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A valid Python math expression (e.g., "2 + 2", "10 * 5")
    
    Returns:
        The result as a string
    """
    try:
        # Only allow basic math operations for safety
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations are allowed"
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time.
    
    Returns:
        Current datetime as a formatted string
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about AI and LangGraph.
    
    Args:
        query: The search query
        
    Returns:
        Relevant information from the knowledge base
    """
    # Simulated knowledge base
    kb = {
        "langgraph": (
            "LangGraph is a library for building stateful, multi-actor applications "
            "with LLMs. It provides StateGraph for defining workflows, ToolNode for "
            "executing tools, and checkpointers for memory persistence."
        ),
        "tools": (
            "Tools in LangGraph allow LLMs to take actions and interact with external "
            "systems. They are defined using the @tool decorator and bound to LLMs "
            "using bind_tools()."
        ),
        "agents": (
            "AI agents are systems that can reason, plan, and execute actions "
            "autonomously. LangGraph provides the building blocks to create "
            "production-ready agents with ReAct-style loops."
        ),
        "memory": (
            "LangGraph supports memory through checkpointers. InMemorySaver is for "
            "development, while SqliteSaver provides persistent storage for production."
        ),
        "react": (
            "ReAct (Reasoning and Acting) is a pattern where the agent alternates "
            "between thinking (reasoning) and taking actions (tool calls) until "
            "it can provide a final answer."
        ),
    }
    
    query_lower = query.lower()
    matches = []
    
    for key, value in kb.items():
        if key in query_lower:
            matches.append(value)
    
    if matches:
        return " ".join(matches)
    
    return "No relevant information found in the knowledge base for: " + query


@tool
def string_length(text: str) -> str:
    """Calculate the length of a string.
    
    Args:
        text: The text to measure
        
    Returns:
        The length of the text
    """
    return f"The text has {len(text)} characters"


# Collect all tools
tools = [calculator, get_current_time, search_knowledge_base, string_length]


# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
def get_llm_with_tools():
    """Create LLM with tools bound."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # Lower temperature for more predictable tool usage
    )
    return llm.bind_tools(tools)


llm_with_tools = get_llm_with_tools()


# -----------------------------------------------------------------------------
# Graph Nodes
# -----------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """The reasoning node - decides whether to call tools or respond.
    
    This node:
    1. Receives the current message history
    2. Invokes the LLM with tools
    3. Returns either:
       - An AIMessage with tool_calls (to be processed by ToolNode)
       - An AIMessage with content (final response)
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Create tool node
tool_node = ToolNode(tools)


# -----------------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------------
def build_agent():
    """Build and compile the agent graph with tools."""
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    
    # Add edges
    graph.add_edge(START, "agent")
    
    # Conditional edge: if tool calls, go to tools; otherwise end
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        }
    )
    
    # After tools execute, go back to agent
    graph.add_edge("tools", "agent")
    
    return graph.compile()


# Create the agent
agent = build_agent()


# -----------------------------------------------------------------------------
# Example Queries
# -----------------------------------------------------------------------------
def run_examples():
    """Run example queries to demonstrate tool usage."""
    
    system_message = SystemMessage(
        content=(
            "You are a helpful AI assistant with access to tools. "
            "Use tools when appropriate to answer questions accurately. "
            "Always explain your reasoning."
        )
    )
    
    examples = [
        # Math calculation
        "What is 125 * 8 + 45?",
        
        # Time query
        "What is the current date and time?",
        
        # Knowledge search
        "What is LangGraph and how does it work?",
        
        # Multi-tool query
        "What time is it and what is 256 / 16?",
        
        # String operation
        "How many characters are in the phrase 'LangGraph is awesome'?",
        
        # No tool needed
        "Hello! How are you?",
    ]
    
    for query in examples:
        print("=" * 70)
        print(f"User: {query}")
        print("-" * 70)
        
        result = agent.invoke({
            "messages": [system_message, HumanMessage(content=query)]
        })
        
        # Display tool calls if any
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[Tool Call] {tc['name']}: {tc['args']}")
        
        # Display final response
        print(f"\nAssistant: {result['messages'][-1].content}")
        print()


def interactive_agent():
    """Run an interactive agent session."""
    print("=" * 70)
    print("Interactive Agent with Tools")
    print("Available tools: calculator, get_current_time, search_knowledge_base, string_length")
    print("Type 'quit', 'exit', or 'q' to end.")
    print("=" * 70)
    
    messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant with access to tools. "
                "Use the available tools when needed to provide accurate answers. "
                "Be concise but thorough in your explanations."
            )
        )
    ]
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        messages.append(HumanMessage(content=user_input))
        
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        
        # Show which tools were called
        for msg in messages[-5:]:  # Check recent messages for tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  [Using tool: {tc['name']}]")
        
        print(f"Assistant: {messages[-1].content}")


# -----------------------------------------------------------------------------
# Visualization Helper
# -----------------------------------------------------------------------------
def visualize_graph():
    """Print graph structure information."""
    print("Agent Graph Structure:")
    print("  START -> agent")
    print("  agent -> tools (if tool_calls present)")
    print("  agent -> END (if no tool_calls)")
    print("  tools -> agent (loop back)")
    print()


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    """Run demonstrations."""
    visualize_graph()
    run_examples()
    
    # Uncomment for interactive mode
    # interactive_agent()


if __name__ == "__main__":
    main()
