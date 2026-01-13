# LangGraph Agentic AI Assistant Tutorial

A comprehensive, hands-on tutorial for building production-ready AI assistants using LangGraph - from basic chatbots to full-featured agents with tools, memory, and streaming.

## Tutorial Overview

This tutorial teaches you to build an agentic AI assistant similar to Claude, Gemini, and ChatGPT. You will learn to create an agent that can:

- Answer questions using LLMs
- Take actions using tools (calculations, web search, etc.)
- Maintain conversation context across turns
- Stream responses in real-time
- Handle errors gracefully

## Prerequisites

- Python 3.10+
- Basic understanding of LangGraph (StateGraph, nodes, edges)
- OpenAI API key or Ollama for local LLMs

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install langgraph langchain-openai langchain-core python-dotenv

# For local LLMs (optional)
pip install langchain-ollama
```

## Modules

| Module | Topic | Description |
|--------|-------|-------------|
| [01](01_introduction.md) | Introduction | Agentic AI concepts and LangGraph fundamentals |
| [02](02_basic_chatbot.md) | Basic Chatbot | Foundation with StateGraph and messaging |
| [03](03_tools.md) | Tools and Actions | Adding tool calling capabilities |
| [04](04_memory.md) | Memory Management | Persistence and conversation history |
| [05](05_advanced_patterns.md) | Advanced Patterns | HITL, multi-agent, error handling |
| [06](06_production.md) | Production Features | Streaming, async, observability |
| [07](07_complete_agent.md) | Complete Agent | Full implementation with UI |

## Module Progression

Each module builds conceptually on the previous one, but code files are self-contained for easier learning:

```
Module 01: Introduction
    Concepts: Agentic AI, ReAct pattern, LangGraph basics
         |
         v
Module 02: Basic Chatbot
    Core: StateGraph, MessagesState, nodes, edges
         |
         v
Module 03: Tools
    Adds: @tool decorator, ToolNode, tools_condition
         |
         v
Module 04: Memory
    Adds: Checkpointers (MemorySaver, SqliteSaver)
         |
         v
Module 05: Advanced Patterns
    Adds: HITL (interrupt), multi-agent, retries
         |
         v
Module 06: Production
    Adds: Streaming, rate limiting, observability
         |
         v
Module 07: Complete Agent
    Combines: ALL concepts from Modules 01-06
    Adds: Streamlit UI, FastAPI server
```

**Note**: Each code file is self-contained so you can run any module independently and copy specific patterns you need.

## Quick Start

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Define a tool
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Create LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([calculate])

# Build the graph
# ... (see Module 03 for full implementation)
```

## Architecture

```
User Input
    |
    v
+-------------------+
|   LangGraph       |
|   StateGraph      |
+-------------------+
    |
    v
+-------------------+     +-------------------+
|   LLM Node        |<--->|   Tool Node       |
|   (Reasoning)     |     |   (Actions)       |
+-------------------+     +-------------------+
    |
    v
+-------------------+
|   Checkpointer    |
|   (Memory)        |
+-------------------+
    |
    v
User Output
```

## Running the Examples

Each module has accompanying code in the `code/` directory:

```bash
# Run basic chatbot
python code/02_basic_chatbot.py

# Run agent with tools
python code/03_tools.py

# Run complete agent
python code/07_complete_agent.py
```

## Author

Senior Staff AI Engineer Tutorial Series
