# Module 05: Advanced Agent Patterns

In this module, you will learn advanced patterns used in production AI systems like Claude, Gemini, and ChatGPT - including human-in-the-loop, multi-agent architectures, error handling, and subgraphs.

## Learning Objectives

By the end of this module, you will:
- Implement human-in-the-loop (HITL) with `interrupt()`
- Build multi-agent systems with supervisors
- Handle errors gracefully with retry patterns
- Create modular agents using subgraphs
- Implement parallel tool execution

## Human-in-the-Loop (HITL)

HITL allows agents to pause execution and wait for human approval or input. This is critical for:

- Sensitive actions (sending emails, making purchases)
- Uncertain decisions requiring expert input
- Quality control checkpoints

### Using interrupt()

```python
from langgraph.types import interrupt

def sensitive_action_node(state: AgentState) -> dict:
    """Node that requires human approval."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if this action needs approval
    if "send email" in last_message.content.lower():
        # Pause execution and wait for human
        approval = interrupt({
            "question": "Do you want to send this email?",
            "context": last_message.content
        })
        
        if approval.get("approved") != True:
            return {"messages": [AIMessage(content="Email sending cancelled.")]}
    
    # Continue with action
    return {"messages": [AIMessage(content="Action completed.")]}
```

### Resuming After Interrupt

```python
from langgraph.types import Command

# Initial invocation - will pause at interrupt
result = agent.invoke({"messages": [...]}, config=config)

# Check if interrupted
if result.get("__interrupt__"):
    print(f"Agent paused: {result['__interrupt__']}")
    
    # Get human approval
    user_approval = input("Approve? (yes/no): ")
    
    # Resume with approval
    result = agent.invoke(
        Command(resume={"approved": user_approval == "yes"}),
        config=config
    )
```

## Multi-Agent with Supervisor Pattern

For complex tasks, use multiple specialized agents coordinated by a supervisor:

```
                    +-------------+
                    | Supervisor  |
                    +-------------+
                          |
           +--------------+--------------+
           |              |              |
           v              v              v
     +---------+    +---------+    +---------+
     | Research|    | Writer  |    | Reviewer|
     | Agent   |    | Agent   |    | Agent   |
     +---------+    +---------+    +---------+
```

### Supervisor Implementation

```python
from typing import Literal

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str

def supervisor_node(state: SupervisorState) -> dict:
    """Decide which agent to invoke next."""
    messages = state["messages"]
    
    # LLM decides next step
    response = supervisor_llm.invoke([
        SystemMessage(content="""
            You are a supervisor coordinating agents.
            Based on the conversation, decide which agent to use next:
            - 'researcher' for information gathering
            - 'writer' for content creation
            - 'reviewer' for quality checks
            - 'FINISH' when task is complete
            
            Respond with just the agent name.
        """),
        *messages
    ])
    
    return {"next_agent": response.content.strip()}


def route_to_agent(state: SupervisorState) -> Literal["researcher", "writer", "reviewer", "__end__"]:
    """Route to the appropriate agent."""
    next_agent = state["next_agent"]
    if next_agent == "FINISH":
        return "__end__"
    return next_agent
```

## Error Handling and Retries

Production agents must handle failures gracefully:

### Retry Decorator

```python
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    delay = base_delay * (2 ** retries)
                    print(f"Error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


@retry_with_backoff(max_retries=3)
def call_external_api(query: str) -> str:
    """Call external API with retry logic."""
    # Your API call here
    pass
```

### Error Handling Node

```python
def error_handler_node(state: AgentState) -> dict:
    """Handle errors gracefully."""
    try:
        # Attempt main logic
        result = process_request(state)
        return {"messages": [AIMessage(content=result)]}
    except RateLimitError:
        return {"messages": [AIMessage(
            content="Service is busy. Please try again in a moment."
        )]}
    except AuthenticationError:
        return {"messages": [AIMessage(
            content="Authentication failed. Please check your credentials."
        )]}
    except Exception as e:
        return {"messages": [AIMessage(
            content=f"An unexpected error occurred. Our team has been notified."
        )]}
```

## Subgraphs for Modularity

Break complex agents into reusable subgraphs:

```python
def build_research_subgraph():
    """Build a reusable research subgraph."""
    graph = StateGraph(ResearchState)
    
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    
    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)
    
    return graph.compile()


def build_main_graph():
    """Build main graph using subgraphs."""
    research_graph = build_research_subgraph()
    
    main_graph = StateGraph(MainState)
    
    # Add subgraph as a node
    main_graph.add_node("research", research_graph)
    main_graph.add_node("respond", respond_node)
    
    main_graph.add_edge(START, "research")
    main_graph.add_edge("research", "respond")
    main_graph.add_edge("respond", END)
    
    return main_graph.compile()
```

## Parallel Tool Execution

Execute multiple tools concurrently for efficiency:

```python
from langgraph.constants import Send

def parallel_tools_node(state: AgentState):
    """Execute multiple tools in parallel."""
    messages = state["messages"]
    last_msg = messages[-1]
    
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 1:
        # Send each tool call to parallel processing
        return [
            Send("execute_tool", {"tool_call": tc})
            for tc in last_msg.tool_calls
        ]
    
    return {"messages": messages}
```

## Complete Code

See [code/05_advanced_patterns.py](code/05_advanced_patterns.py) for the complete working examples.

## Best Practices Summary

| Pattern | When to Use |
|---------|-------------|
| HITL | Sensitive/irreversible actions |
| Multi-Agent | Complex, multi-step tasks |
| Subgraphs | Reusable components |
| Retry | External API calls |
| Parallel | Multiple independent tools |

## Exercises

1. **HITL Email Agent**: Build an agent that drafts and sends emails with approval
2. **Research Assistant**: Create a multi-agent system with researcher and writer
3. **Fault Tolerant**: Add retry logic to the tools from Module 03

## Next Steps

In [Module 06](06_production.md), we will cover production-ready features: streaming, async patterns, and observability.

---

[Back to README](README.md) | [Previous: Memory](04_memory.md) | [Next: Production](06_production.md)
