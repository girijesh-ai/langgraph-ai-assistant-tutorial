# Module 04: Memory and State Management

In this module, you will add memory to your agent, enabling it to persist conversations across sessions and maintain context in multi-turn interactions - just like Claude, Gemini, and ChatGPT.

## Learning Objectives

By the end of this module, you will:
- Understand LangGraph checkpointing system
- Use `MemorySaver` for development
- Use `SqliteSaver` for persistent storage
- Manage multiple conversation threads
- Implement conversation history management

## Why Memory Matters

Without memory, agents are stateless - they forget everything after each invocation. Memory enables:

1. **Multi-turn conversations**: Agent remembers what you said earlier
2. **User preferences**: Agent recalls your settings
3. **Task continuity**: Resume interrupted workflows
4. **Personalization**: Learn from past interactions

## Checkpointers in LangGraph

LangGraph uses **checkpointers** to save and restore state:

| Checkpointer | Use Case | Persistence |
|--------------|----------|-------------|
| `MemorySaver` | Development/Testing | In-process memory (lost on restart) |
| `SqliteSaver` | Local persistence | SQLite database file |
| `PostgresSaver` | Production | PostgreSQL database |

## Using MemorySaver (Development)

Perfect for development and testing:

```python
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer
memory = MemorySaver()

# Compile graph with checkpointer
agent = graph.compile(checkpointer=memory)
```

## The Thread Concept

Each conversation is identified by a **thread_id**. Different thread IDs create separate conversation contexts:

```python
# Thread 1 - User Alice
config_1 = {"configurable": {"thread_id": "alice-123"}}

# Thread 2 - User Bob  
config_2 = {"configurable": {"thread_id": "bob-456"}}

# Each thread has its own message history
agent.invoke({"messages": [...]}, config=config_1)  # Alice's conversation
agent.invoke({"messages": [...]}, config=config_2)  # Bob's conversation
```

## Multi-Turn Conversation with Memory

With checkpointing, you do not need to manually track message history:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "session-1"}}

# Turn 1
agent.invoke(
    {"messages": [HumanMessage(content="My favorite color is blue")]},
    config=config
)

# Turn 2 - Agent remembers!
result = agent.invoke(
    {"messages": [HumanMessage(content="What is my favorite color?")]},
    config=config
)

print(result["messages"][-1].content)
# Output: Your favorite color is blue.
```

## Using SqliteSaver (Persistent)

For persistence across application restarts:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create persistent checkpointer
with SqliteSaver.from_conn_string("checkpoints.db") as memory:
    agent = graph.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "persistent-session"}}
    
    # This conversation survives restarts
    result = agent.invoke({"messages": [...]}, config=config)
```

### Async SqliteSaver

For async applications:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
    agent = graph.compile(checkpointer=memory)
    result = await agent.ainvoke({"messages": [...]}, config=config)
```

## Inspecting State

You can inspect and manipulate saved state:

```python
# Get current state
state = agent.get_state(config)
print(state.values)  # Current state values
print(state.next)    # Next node to execute (if any)

# Get state history
for state in agent.get_state_history(config):
    print(f"Step {state.metadata['step']}: {len(state.values['messages'])} messages")
```

## Updating State

Modify state externally (useful for corrections):

```python
# Add a message externally
agent.update_state(
    config,
    {"messages": [HumanMessage(content="Actually, my favorite color is red")]}
)
```

## Managing Conversation Length

Conversations can grow large. Manage message history:

```python
def trim_messages(state: AgentState) -> dict:
    """Keep only the last N messages to manage context window."""
    messages = state["messages"]
    
    # Keep system message + last 10 messages
    if len(messages) > 11:
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        messages = system_msgs + other_msgs[-10:]
    
    return {"messages": messages}
```

Or use the built-in message trimmer:

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    messages,
    max_tokens=1000,
    token_counter=llm,
    strategy="last",  # Keep last messages
)
```

## Complete Architecture with Memory

```
User Input
    |
    v
+-------------------+
|   Config with     |
|   thread_id       |
+-------------------+
    |
    v
+-------------------+
|   Agent Graph     |
+-------------------+
    |
    v
+-------------------+
|   Checkpointer    |<--- Saves/Restores State
|   (Memory/SQLite) |
+-------------------+
    |
    v
+-------------------+
|   Persistent      |
|   Storage         |
+-------------------+
```

## Complete Code

See [code/04_memory.py](code/04_memory.py) for the complete working example.

## Common Issues

### Issue: Conversation Not Persisting
- Ensure you are using the same `thread_id`
- For `SqliteSaver`, check database file is being created
- Verify checkpointer is passed to `compile()`

### Issue: Context Window Exceeded
- Implement message trimming
- Use summarization for long conversations

## Exercises

1. **Multi-User Chat**: Create an agent that handles multiple users with separate threads
2. **Session Recovery**: Use SqliteSaver and verify conversation persists after restart
3. **State Inspection**: Build a debug tool to view conversation history

## Next Steps

In [Module 05](05_advanced_patterns.md), we will explore advanced patterns like human-in-the-loop, multi-agent systems, and error handling.

---

[Back to README](README.md) | [Previous: Tools](03_tools.md) | [Next: Advanced Patterns](05_advanced_patterns.md)
