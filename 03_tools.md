# Module 03: Adding Tools and Tool Calling

In this module, you will add tool calling capabilities to your agent. This is what separates a simple chatbot from an **agentic AI assistant** - the ability to take actions in the world.

## Prerequisites

Before starting this module, you should have completed:
- ✅ **Module 02**: Understand StateGraph, nodes, edges, and message flow
- ✅ Basic Python: Functions, decorators, type hints

**Note:** This module shows tools WITHOUT memory. In Module 04, you'll learn how to persist tool-using conversations across sessions.

## Learning Objectives

By the end of this module, you will:
- Create custom tools using the `@tool` decorator
- Bind tools to LLMs using `bind_tools`
- Use `ToolNode` for automatic tool execution
- Implement conditional routing with `tools_condition`
- Build a ReAct-style agent loop

## What Makes an Agent "Agentic"

The key difference between a chatbot and an agent:

| Chatbot | Agent |
|---------|-------|
| Answers questions | Answers AND takes actions |
| Single LLM call | Multiple reasoning cycles |
| No external effects | Can modify state, call APIs |
| Passive | Proactive |

## Creating Tools

Tools are functions that the LLM can call. Use the `@tool` decorator to define them:

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A valid Python math expression (e.g., "2 + 2", "10 * 5")
    
    Returns:
        The result as a string
    """
    try:
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
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool  
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information.
    
    Args:
        query: The search query
        
    Returns:
        Relevant information from the knowledge base
    """
    # Simulated knowledge base
    kb = {
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "tools": "Tools allow LLMs to take actions and interact with external systems.",
        "agents": "AI agents are systems that can reason, plan, and execute actions autonomously."
    }
    
    for key, value in kb.items():
        if key.lower() in query.lower():
            return value
    
    return "No relevant information found in the knowledge base."
```

### Tool Best Practices

1. **Clear Docstrings**: The LLM uses docstrings to understand when to call the tool
2. **Type Hints**: Always include type hints for arguments
3. **Error Handling**: Return helpful error messages, do not raise exceptions
4. **Descriptive Names**: Use verb-noun naming (e.g., `get_weather`, `send_email`)

## Binding Tools to the LLM

Use `bind_tools` to attach tools to your LLM:

```python
from langchain_openai import ChatOpenAI

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind tools
tools = [calculator, get_current_time, search_knowledge_base]
llm_with_tools = llm.bind_tools(tools)
```

## The ToolNode

LangGraph provides `ToolNode` to automatically execute tool calls:

```python
from langgraph.prebuilt import ToolNode

# Create tool node with our tools
tool_node = ToolNode(tools)
```

### How ToolNode Works

1. Receives state with AIMessage containing tool_calls
2. Executes each tool call
3. Returns ToolMessage(s) with results
4. These messages get added to state

## Conditional Routing with tools_condition

The `tools_condition` function checks if the LLM wants to call tools:

```python
from langgraph.prebuilt import tools_condition

# Returns "tools" if tool_calls present, otherwise END
```

This enables the ReAct loop:
- LLM decides whether to call a tool or respond
- If tool call: execute tool and loop back
- If no tool call: end and return response

## Building the Agent Graph

First, define the state (same pattern from Module 02):

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """State for the agent with tools."""
    messages: Annotated[list[BaseMessage], add_messages]
```

Now build the graph:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def agent_node(state: ChatState) -> dict:
    """The reasoning node - decides whether to call tools or respond."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build graph
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "agent")

# Conditional edge: agent -> tools OR agent -> END
graph.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",  # If tool calls present
        "__end__": END     # If no tool calls
    }
)

# Tool results go back to agent for processing
graph.add_edge("tools", "agent")

# Compile
agent = graph.compile()
```

### ReAct Loop Visualization

```
    START
      |
      v
  +--------+
  | agent  |<---------+
  +--------+          |
      |               |
      v               |
  tools_condition     |
    /    \            |
   v      v           |
tools    END          |
  |                   |
  +-------------------+
```

## Using the Agent

```python
from langchain_core.messages import HumanMessage

# Query requiring calculation
result = agent.invoke({
    "messages": [HumanMessage(content="What is 25 * 4 + 10?")]
})

# The agent will:
# 1. Recognize this needs calculation
# 2. Call the calculator tool
# 3. Return the result with explanation

print(result["messages"][-1].content)
```

## Multi-Tool Queries

Agents can use multiple tools to answer complex queries:

```python
result = agent.invoke({
    "messages": [HumanMessage(
        content="What time is it and what is 100 divided by 4?"
    )]
})
```

The agent may call both `get_current_time` and `calculator` tools.

## Complete Code

See [code/03_tools.py](code/03_tools.py) for the complete working example.

## Debugging Tools

To see what tools the LLM is calling:

```python
# Check the second-to-last message for tool calls
for msg in result["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"Tool calls: {msg.tool_calls}")
```

## Common Issues

### Issue: LLM Not Calling Tools
- Check docstrings are descriptive
- Try more explicit prompts ("calculate", "search for")
- Reduce temperature to 0 for more deterministic behavior

### Issue: Tool Execution Errors
- Ensure tool node has all tools that LLM was bound with
- Check tool is returning strings (not other types)

## When NOT to Use Tools

Tools add complexity and latency. **Skip tools** when:

| Scenario | Better Approach |
|----------|-----------------|
| Simple Q&A | Direct LLM response |
| Information already in prompt | Use context instead |
| Real-time conversation | Tools add latency |
| User just wants to chat | Let LLM be conversational |

**Example:** "What's 2+2?" - LLM can answer directly, doesn't need calculator tool.

**Rule of thumb:** If the answer requires external data or computation, use tools. If it's knowledge or reasoning, let LLM handle it.

## Bridging to Module 04

**What you've built:** An agent that uses tools to take actions.

**The problem:** Your agent forgets everything after each `invoke()`. Try this:

```python
# Turn 1
agent.invoke({"messages": [HumanMessage("Calculate 50 * 3")]})
# Agent: The result is 150

# Turn 2  
agent.invoke({"messages": [HumanMessage("What was my last calculation?")]})
# Agent: I don't have information about your previous calculations
```

**The agent forgot!** In Module 04, you'll add memory so agents remember conversations across invocations - just like ChatGPT does.

## Exercises

1. **Weather Tool**: Create a mock weather tool that returns forecast
2. **Multiple Parameters**: Create a tool that takes multiple arguments
3. **Tool Selection**: Write queries that specifically trigger each tool

## Next Steps

In [Module 04](04_memory.md), we will add memory to persist conversations across sessions.

---

[Back to README](README.md) | [Previous: Basic Chatbot](02_basic_chatbot.md) | [Next: Memory](04_memory.md)
