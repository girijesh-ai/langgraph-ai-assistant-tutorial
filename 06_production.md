# Module 06: Building Production-Ready Features

In this module, you will add production-ready features to your agent: streaming responses, async patterns, observability, and security - the features that make Claude, Gemini, and ChatGPT feel smooth and responsive.

## Learning Objectives

By the end of this module, you will:
- Implement token-by-token streaming
- Use async patterns for better performance
- Add observability with Langfuse
- Handle rate limiting and timeouts
- Implement security best practices

## Streaming Responses

Streaming provides real-time feedback as the LLM generates tokens:

### Basic Streaming with astream_events

```python
async def stream_response(agent, messages: list, config: dict):
    """Stream response tokens to the user."""
    print("Assistant: ", end="", flush=True)
    
    async for event in agent.astream_events(
        {"messages": messages},
        config=config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
    
    print()  # Newline after response
```

### Event Types

| Event | Description |
|-------|-------------|
| `on_chat_model_start` | LLM invocation begins |
| `on_chat_model_stream` | Token chunk received |
| `on_chat_model_end` | LLM invocation complete |
| `on_tool_start` | Tool execution begins |
| `on_tool_end` | Tool execution complete |
| `on_chain_start` | Graph node begins |
| `on_chain_end` | Graph node complete |

### Streaming with Tool Calls

```python
async def stream_with_tools(agent, messages: list, config: dict):
    """Stream with visibility into tool usage."""
    
    async for event in agent.astream_events(
        {"messages": messages},
        config=config,
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
                
        elif kind == "on_tool_start":
            tool_name = event["name"]
            print(f"\n[Using tool: {tool_name}]", flush=True)
            
        elif kind == "on_tool_end":
            tool_output = event["data"]["output"]
            print(f"[Tool result: {tool_output[:50]}...]", flush=True)
    
    print()
```

## Async Patterns

For production applications, use async throughout:

### Async Node Definition

```python
async def async_agent_node(state: AgentState) -> dict:
    """Async agent node for better concurrency."""
    messages = state["messages"]
    
    # Use ainvoke for async LLM calls
    response = await llm_with_tools.ainvoke(messages)
    
    return {"messages": [response]}
```

### Async Graph Invocation

```python
async def main():
    # Build graph
    agent = build_agent()
    
    # Async invocation
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Hello")]},
        config={"configurable": {"thread_id": "async-session"}}
    )
    
    return result

# Run
import asyncio
asyncio.run(main())
```

## Observability with Langfuse

Track, debug, and analyze your agent in production:

### Basic Setup

```python
from langfuse.callback import CallbackHandler

# Create Langfuse handler
langfuse_handler = CallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

# Add to config
config = {
    "configurable": {"thread_id": "session-1"},
    "callbacks": [langfuse_handler]
}

result = agent.invoke({"messages": [...]}, config=config)
```

### What Langfuse Tracks

- Full conversation traces
- Token usage and costs
- Latency per node
- Tool call success/failure
- Custom scores and feedback

## Rate Limiting

Protect your agent from abuse:

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window = timedelta(seconds=window_seconds)
        self.calls = defaultdict(list)
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user can make a call."""
        now = datetime.now()
        cutoff = now - self.window
        
        # Clean old calls
        self.calls[user_id] = [
            t for t in self.calls[user_id] if t > cutoff
        ]
        
        if len(self.calls[user_id]) >= self.max_calls:
            return False
        
        self.calls[user_id].append(now)
        return True


# Usage
limiter = RateLimiter(max_calls=10, window_seconds=60)

def rate_limited_invoke(agent, messages, user_id, config):
    """Invoke agent with rate limiting."""
    if not limiter.is_allowed(user_id):
        raise Exception("Rate limit exceeded. Please wait before trying again.")
    
    return agent.invoke({"messages": messages}, config=config)
```

## Timeouts

Prevent runaway requests:

```python
import asyncio

async def invoke_with_timeout(agent, messages, config, timeout_seconds=30):
    """Invoke agent with timeout."""
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": messages}, config=config),
            timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        return {
            "messages": [AIMessage(
                content="Request timed out. Please try a simpler query."
            )]
        }
```

## Security Best Practices

### Input Validation

```python
import re

def validate_input(user_input: str) -> tuple[bool, str]:
    """Validate and sanitize user input."""
    # Check length
    if len(user_input) > 4000:
        return False, "Input too long. Maximum 4000 characters."
    
    # Check for prompt injection attempts
    injection_patterns = [
        r"ignore previous instructions",
        r"disregard all prior",
        r"new system prompt",
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, user_input.lower()):
            return False, "Invalid input detected."
    
    return True, user_input
```

### Tool Sandboxing

```python
@tool
def safe_calculator(expression: str) -> str:
    """Safe calculator that only allows basic operations."""
    # Whitelist allowed characters
    allowed = set("0123456789+-*/.() ")
    
    if not all(c in allowed for c in expression):
        return "Error: Only basic math operations allowed"
    
    # Limit expression length
    if len(expression) > 100:
        return "Error: Expression too long"
    
    try:
        result = eval(expression)
        return str(result)
    except Exception:
        return "Error: Invalid expression"
```

## Complete Code

See [code/06_production.py](code/06_production.py) for the complete working example.

## Production Checklist

Before deploying your agent:

- [ ] Streaming enabled for responsive UX
- [ ] Async patterns for scalability
- [ ] Observability configured (Langfuse/LangSmith)
- [ ] Rate limiting implemented
- [ ] Request timeouts configured
- [ ] Input validation in place
- [ ] Error handling comprehensive
- [ ] Logging structured and complete
- [ ] Secrets managed securely
- [ ] Cost controls in place

## Exercises

1. **Full Streaming**: Implement streaming that shows both tokens and tool usage
2. **Custom Metrics**: Add custom metrics to Langfuse for business-specific tracking
3. **Load Testing**: Use a tool like locust to test agent under load

## Next Steps

In [Module 07](07_complete_agent.md), we will put everything together into a complete, production-ready agent with a Streamlit UI.

---

[Back to README](README.md) | [Previous: Advanced Patterns](05_advanced_patterns.md) | [Next: Complete Agent](07_complete_agent.md)
