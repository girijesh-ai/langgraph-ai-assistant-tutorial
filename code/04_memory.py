"""
Module 04: Memory and State Management
=======================================
A LangGraph agent with persistent memory:
- MemorySaver for development
- SqliteSaver for production
- Multi-thread conversation management
- State inspection and manipulation
"""

import os
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
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
    """State for the agent with memory."""
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------------------------------------------------------
# Tool Definitions
# -----------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A valid Python math expression
    
    Returns:
        The result as a string
    """
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations are allowed"
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
def remember_preference(key: str, value: str) -> str:
    """Remember a user preference.
    
    Args:
        key: The preference name (e.g., 'favorite_color')
        value: The preference value (e.g., 'blue')
    
    Returns:
        Confirmation message
    """
    return f"I will remember that your {key} is {value}."


tools = [calculator, get_current_time, remember_preference]


# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# -----------------------------------------------------------------------------
# Graph Nodes
# -----------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """The reasoning node with tools."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# -----------------------------------------------------------------------------
# Graph Construction with Memory
# -----------------------------------------------------------------------------
def build_agent_with_memory(checkpointer=None):
    """Build agent with optional checkpointer."""
    graph = StateGraph(AgentState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")
    
    return graph.compile(checkpointer=checkpointer)


# -----------------------------------------------------------------------------
# Memory Demonstrations
# -----------------------------------------------------------------------------
def demo_without_memory():
    """Show agent without memory - forgets between calls."""
    print("=" * 70)
    print("Demo: Agent WITHOUT Memory")
    print("=" * 70)
    
    agent = build_agent_with_memory(checkpointer=None)
    
    # Turn 1
    print("\nTurn 1:")
    result = agent.invoke({
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="My name is Alice and I love Python.")
        ]
    })
    print(f"User: My name is Alice and I love Python.")
    print(f"Assistant: {result['messages'][-1].content}")
    
    # Turn 2 - Agent forgets!
    print("\nTurn 2:")
    result = agent.invoke({
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is my name?")
        ]
    })
    print(f"User: What is my name?")
    print(f"Assistant: {result['messages'][-1].content}")
    print("\n[Notice: Agent does not remember the name!]\n")


def demo_with_memory():
    """Show agent with memory - remembers across calls."""
    print("=" * 70)
    print("Demo: Agent WITH Memory (MemorySaver)")
    print("=" * 70)
    
    # Create memory checkpointer
    memory = MemorySaver()
    agent = build_agent_with_memory(checkpointer=memory)
    
    # Configuration with thread_id
    config = {"configurable": {"thread_id": "demo-session-1"}}
    
    # Turn 1
    print("\nTurn 1:")
    result = agent.invoke({
        "messages": [
            SystemMessage(content="You are a helpful assistant. Remember user information."),
            HumanMessage(content="My name is Bob and I work as a data scientist.")
        ]
    }, config=config)
    print(f"User: My name is Bob and I work as a data scientist.")
    print(f"Assistant: {result['messages'][-1].content}")
    
    # Turn 2 - Agent remembers!
    print("\nTurn 2:")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is my name and job?")]
    }, config=config)
    print(f"User: What is my name and job?")
    print(f"Assistant: {result['messages'][-1].content}")
    
    # Turn 3 - Continue conversation
    print("\nTurn 3:")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is 100 * 3.14?")]
    }, config=config)
    print(f"User: What is 100 * 3.14?")
    print(f"Assistant: {result['messages'][-1].content}")
    
    print("\n[Notice: Agent remembers everything across turns!]\n")
    
    return agent, config


def demo_multiple_threads():
    """Show separate conversations with different thread_ids."""
    print("=" * 70)
    print("Demo: Multiple Conversation Threads")
    print("=" * 70)
    
    memory = MemorySaver()
    agent = build_agent_with_memory(checkpointer=memory)
    
    system_msg = SystemMessage(content="You are a helpful assistant.")
    
    # Thread 1 - Alice
    config_alice = {"configurable": {"thread_id": "alice-thread"}}
    agent.invoke({
        "messages": [system_msg, HumanMessage(content="I am Alice, I love cats.")]
    }, config=config_alice)
    
    # Thread 2 - Bob
    config_bob = {"configurable": {"thread_id": "bob-thread"}}
    agent.invoke({
        "messages": [system_msg, HumanMessage(content="I am Bob, I love dogs.")]
    }, config=config_bob)
    
    # Query each thread
    print("\nQuerying Alice's thread:")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is my name and pet preference?")]
    }, config=config_alice)
    print(f"Assistant: {result['messages'][-1].content}")
    
    print("\nQuerying Bob's thread:")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is my name and pet preference?")]
    }, config=config_bob)
    print(f"Assistant: {result['messages'][-1].content}")
    
    print("\n[Notice: Each thread has separate memory!]\n")


def demo_state_inspection(agent, config):
    """Show how to inspect and manipulate state."""
    print("=" * 70)
    print("Demo: State Inspection")
    print("=" * 70)
    
    # Get current state
    state = agent.get_state(config)
    
    print(f"\nCurrent state has {len(state.values.get('messages', []))} messages")
    print(f"Next node(s): {state.next}")
    print(f"Metadata: {state.metadata}")
    
    # Show message summary
    print("\nMessage history:")
    for i, msg in enumerate(state.values.get("messages", [])):
        msg_type = type(msg).__name__
        content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
        print(f"  {i+1}. [{msg_type}] {content_preview}")
    
    print()


def demo_sqlite_persistence():
    """Demonstrate SQLite persistence (commented for safety)."""
    print("=" * 70)
    print("Demo: SQLite Persistence (Code Example)")
    print("=" * 70)
    
    print("""
To use SQLite for persistent memory across restarts:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create persistent checkpointer
with SqliteSaver.from_conn_string("agent_memory.db") as checkpointer:
    agent = build_agent_with_memory(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "persistent-user"}}
    
    # This conversation survives application restarts
    result = agent.invoke({
        "messages": [HumanMessage(content="Remember this!")]
    }, config=config)
```

For async applications:
```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("agent_memory.db") as checkpointer:
    agent = build_agent_with_memory(checkpointer=checkpointer)
    result = await agent.ainvoke({"messages": [...]}, config=config)
```
""")


def interactive_with_memory():
    """Run interactive session with memory."""
    print("=" * 70)
    print("Interactive Agent with Memory")
    print("Type 'quit' to exit, 'history' to see conversation, 'new' for new thread")
    print("=" * 70)
    
    memory = MemorySaver()
    agent = build_agent_with_memory(checkpointer=memory)
    
    thread_id = "interactive-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initial system message
    agent.invoke({
        "messages": [SystemMessage(
            content="You are a helpful AI assistant with memory. Remember what the user tells you."
        )]
    }, config=config)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
            
        if user_input.lower() == "history":
            state = agent.get_state(config)
            print(f"\n[History: {len(state.values.get('messages', []))} messages in thread '{thread_id}']")
            for msg in state.values.get("messages", []):
                print(f"  - {type(msg).__name__}: {str(msg.content)[:60]}...")
            continue
            
        if user_input.lower() == "new":
            thread_id = f"interactive-{datetime.now().timestamp()}"
            config = {"configurable": {"thread_id": thread_id}}
            print(f"[Started new thread: {thread_id}]")
            continue
        
        result = agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)
        
        print(f"Assistant: {result['messages'][-1].content}")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    """Run memory demonstrations."""
    # Demo 1: Without memory
    demo_without_memory()
    
    # Demo 2: With memory
    agent, config = demo_with_memory()
    
    # Demo 3: Multiple threads
    demo_multiple_threads()
    
    # Demo 4: State inspection
    demo_state_inspection(agent, config)
    
    # Demo 5: SQLite info
    demo_sqlite_persistence()
    
    # Interactive mode (uncomment to use)
    # interactive_with_memory()


if __name__ == "__main__":
    main()
