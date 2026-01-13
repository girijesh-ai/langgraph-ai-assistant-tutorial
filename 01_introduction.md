# Module 01: Introduction to Agentic AI and LangGraph

Welcome to this tutorial series on building AI assistants with LangGraph. This module introduces the core concepts you need to understand before building your first agent.

## Learning Objectives

By the end of this module, you will understand:
- What makes AI "agentic"
- The ReAct (Reasoning + Acting) pattern
- Core LangGraph concepts: graphs, nodes, edges, state
- When to use LangGraph vs plain LangChain

## What is Agentic AI?

**Agentic AI** refers to AI systems that can autonomously reason, plan, and take actions to accomplish goals. Unlike traditional chatbots that simply respond to prompts, agents can:

| Traditional Chatbot | Agentic AI |
|---------------------|------------|
| Single response per query | Multiple reasoning steps |
| No external actions | Uses tools to interact with world |
| Stateless | Maintains context and memory |
| Reactive only | Proactive and goal-oriented |

### Real-World Examples

Modern AI assistants like Claude, Gemini, and ChatGPT are agentic:
- They can search the web for information
- Execute code to solve problems
- Analyze files and images
- Chain multiple actions to complete complex tasks

## The ReAct Pattern

**ReAct** (Reasoning + Acting) is the foundational pattern for building agents:

```
User Query
    |
    v
+------------------+
|    REASONING     |  <-- "I need to calculate this..."
+------------------+
    |
    v
+------------------+
|     ACTING       |  <-- Calls calculator tool
+------------------+
    |
    v
+------------------+
|   OBSERVATION    |  <-- Tool returns result
+------------------+
    |
    v
(Loop back to REASONING or respond)
```

The agent alternates between:
1. **Reasoning**: Thinking about what to do next
2. **Acting**: Using a tool or taking an action
3. **Observing**: Processing the result

This loop continues until the agent can provide a final answer.

## Introduction to LangGraph

**LangGraph** is a library for building stateful, multi-actor applications with LLMs using a graph-based approach.

### Why LangGraph?

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| Architecture | Chains (linear) | Graphs (flexible) |
| State Management | Manual | Built-in |
| Cycles/Loops | Difficult | Native support |
| Human-in-the-Loop | Manual | Built-in |
| Persistence | Manual | Checkpointers |

Use LangGraph when you need:
- Complex, non-linear workflows
- Cycles in your agent logic (ReAct pattern)
- Built-in persistence and state management
- Human-in-the-loop capabilities

### Core Concepts

#### 1. StateGraph

The foundation of every LangGraph application. It defines:
- The structure of your workflow
- What data flows through the graph (State)

```python
from langgraph.graph import StateGraph

class MyState(TypedDict):
    messages: list  # Data that flows through the graph

graph = StateGraph(MyState)
```

#### 2. Nodes

Functions that process state and return updates:

```python
def my_node(state: MyState) -> dict:
    # Process state
    result = do_something(state["messages"])
    # Return updates to state
    return {"messages": [result]}

graph.add_node("my_node", my_node)
```

#### 3. Edges

Connections between nodes that define flow:

```python
from langgraph.graph import START, END

# Simple edge: A -> B
graph.add_edge("node_a", "node_b")

# Entry and exit
graph.add_edge(START, "first_node")
graph.add_edge("last_node", END)
```

#### 4. Conditional Edges

Dynamic routing based on state:

```python
def route_function(state: MyState) -> str:
    if needs_tool(state):
        return "tool_node"
    return "end"

graph.add_conditional_edges("agent", route_function)
```

#### 5. Compilation

Convert the graph definition into a runnable application:

```python
app = graph.compile()
result = app.invoke({"messages": ["Hello"]})
```

## Visual Architecture

```
+-------------------+
|   StateGraph      |
|   (Blueprint)     |
+-------------------+
         |
         | compile()
         v
+-------------------+
| CompiledGraph     |
| (Runnable App)    |
+-------------------+
         |
         | invoke() / stream()
         v
+-------------------+
|   Execution       |
|   START -> ... -> END
+-------------------+
```

## The Agent Loop

The typical agentic pattern in LangGraph:

```
START
  |
  v
+--------+
| Agent  |<--------+
| (LLM)  |         |
+--------+         |
  |                |
  v                |
[Decision]         |
  |      \         |
  v       v        |
END    Tools ------+
       Node
```

1. **Agent node**: LLM reasons about what to do
2. **Decision point**: Check if tools are needed
3. **Tools node**: Execute requested tools
4. **Loop back**: Return to agent with tool results
5. **End**: When no more tools needed

## Key Terminology

| Term | Definition |
|------|------------|
| **State** | Data that flows through the graph |
| **Node** | A function that processes state |
| **Edge** | Connection between nodes |
| **Checkpointer** | Saves state for persistence/recovery |
| **Thread** | Unique conversation identifier |
| **Tool** | External function the LLM can call |
| **ToolNode** | Built-in node for executing tools |

## Prerequisites for This Tutorial

Before continuing, ensure you have:

1. **Python 3.10+** installed
2. **Basic Python knowledge** (functions, classes, async)
3. **Understanding of LLMs** (prompts, tokens, temperature)
4. **API key** for OpenAI or Ollama installed locally

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install langgraph langchain-openai langchain-core python-dotenv

# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## What You Will Build

By the end of this tutorial series, you will build an AI assistant with:

- **Tools**: Calculator, knowledge search, time
- **Memory**: Persistent conversations across sessions
- **Streaming**: Real-time token delivery
- **Human-in-the-Loop**: Approval for sensitive actions
- **Production features**: Rate limiting, validation
- **UI**: Streamlit chat interface

## Next Steps

Ready to build? Continue to [Module 02: Basic Chatbot](02_basic_chatbot.md) to create your first LangGraph application.

---

[Back to README](README.md) | [Next: Basic Chatbot](02_basic_chatbot.md)
