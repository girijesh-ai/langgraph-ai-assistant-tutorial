# Troubleshooting Guide

Common issues and solutions when building LangGraph agents.

---

## Module 02: Basic Chatbot

### Issue: "Graph has no entrypoint"

**Error:**
```
ValueError: Graph must have an entrypoint: add at least one edge from START to another node
```

**Cause:** Missing `graph.add_edge(START, "node_name")`

**Solution:**
```python
graph.add_edge(START, "chat")  # Add this!
graph.add_conditional_edges(...)
```

### Issue: Messages not accumulating

**Symptom:** Agent forgets previous messages in same conversation

**Cause:** Not using `add_messages` reducer

**Solution:**
```python
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # ← Need this
```

---

## Module 03: Tools

### Issue: LLM not calling tools

**Symptoms:**
- Agent answers directly instead of using calculator
- Tools defined but never invoked

**Common causes:**

1. **Tools not bound to LLM:**
```python
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)  # ← Did you bind?
```

2. **Unclear tool docstrings:**
```python
# Bad
@tool
def calc(x: str) -> str:
    """Do math."""  # Too vague!

# Good  
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression like '2 + 2' or '10 * 5'."""
```

3. **Temperature too high:**
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # ← Use 0 for tools
```

### Issue: Tool called in infinite loop

**Symptom:** Agent calls same tool repeatedly

**Cause:** Tool returns unhelpful response

**Solution:**
```python
@tool
def calculator(expression: str) -> str:
    """Calculate math expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"  # ← Be explicit!
    except Exception as e:
        return f"Error: {str(e)}. Please provide a valid expression."  # ← Help agent
```

### Issue: ToolNode vs tools_condition confusion

**Remember:**
- `ToolNode` = Executes the tools
- `tools_condition` = Decides whether to call tools

```python
# Both are needed!
graph.add_node("tools", ToolNode(tools))  # Executor
graph.add_conditional_edges("agent", tools_condition, ...)  # Decision
```

---

## Module 04: Memory

### Issue: Conversation not persisting

**Symptoms:**
- Agent forgets after each invoke
- "What's my name?" doesn't work

**Checklist:**
1. ✓ Checkpointer passed to compile?
```python
memory = MemorySaver()
agent = graph.compile(checkpointer=memory)  # ← Need this
```

2. ✓ Using config with thread_id?
```python
config = {"configurable": {"thread_id": "session-1"}}  # ← Need this
agent.invoke({"messages": [...]}, config=config)
```

3. ✓ Same thread_id across calls?
```python
# Wrong - different thread_id each time
config = {"configurable": {"thread_id": f"session-{time.time()}"}}

# Right - consistent thread_id
thread_id = "user-123"
config = {"configurable": {"thread_id": thread_id}}
```

### Issue: "No checkpoint found"

**Cause:** Using thread_id that doesn't exist yet

**Solution:** First invoke creates the checkpoint:
```python
# First call - creates checkpoint
agent.invoke({"messages": [SystemMessage(content="...")]}, config=config)

# Subsequent calls - use existing checkpoint
agent.invoke({"messages": [HumanMessage(content="...")]}, config=config)
```

### Issue: Context window exceeded

**Symptom:**
```
Error: maximum context length exceeded
```

**Solution 1: Trim messages**
```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    state["messages"],
    max_tokens=4000,
    strategy="last",
    token_counter=llm
)
```

**Solution 2: Shorter thread lifetimes**
```python
# Instead of one thread per user forever:
thread_id = f"user-{user_id}"

# Use time-based threads:
from datetime import datetime
thread_id = f"user-{user_id}-{datetime.now().strftime('%Y-%m-%d')}"
```

### Issue: SqliteSaver database locked

**Cause:** Multiple processes accessing same SQLite file

**Solutions:**
1. Use in-memory for development:
```python
memory = MemorySaver()  # No DB file
```

2. Use PostgreSQL for production:
```python
from langgraph.checkpoint.postgres import PostgresSaver
```

---

## Module 05: Advanced Patterns

### Issue: interrupt() not pausing execution

**Cause:** Not checking for interrupt in state

**Solution:**
```python
result = agent.invoke({"messages": [...]}, config=config)

# Check if interrupted!
if "__interrupt__" in result:
    print("Paused:", result["__interrupt__"])
    # Handle approval...
```

### Issue: Multi-agent infinite loop

**Symptom:** Supervisor keeps routing between agents forever

**Solution:** Add max iterations:
```python
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    iteration: int  # ← Add counter

def supervisor_node(state: SupervisorState) -> dict:
    iteration = state.get("iteration", 0) + 1
    
    if iteration > 10:  # ← Safety valve
        return {"next_agent": "FINISH", "iteration": iteration}
    
    # Normal routing logic...
    return {"next_agent": next_agent, "iteration": iteration}
```

### Issue: Retry decorator not working

**Cause:** Forgetting to actually call the decorated function

**Wrong:**
```python
@retry_with_backoff(max_retries=3)
def api_call():
    pass

# Forgot to call it!
```

**Right:**
```python
@retry_with_backoff(max_retries=3)
def api_call():
    pass

result = api_call()  # ← Actually call it!
```

---

## Module 06: Production

### Issue: Streaming not working

**Cause 1:** Not using `astream_events`
```python
# Wrong
for chunk in agent.stream({...}):  # Different API

# Right
async for event in agent.astream_events({...}, version="v2"):
```

**Cause 2:** Wrong event type
```python
if event["event"] == "on_chat_model_stream":  # ← Check event type
    chunk = event["data"]["chunk"]
```

### Issue: Rate limiter not blocking requests

**Cause:** Different user_id each time

**Wrong:**
```python
limiter.check(f"user-{time.time()}")  # New ID every time!
```

**Right:**
```python
limiter.check(f"user-{user_id}")  # Consistent ID
```

---

## Module 07: Complete Agent

### Issue: Streamlit UI not updating

**Cause:** Trying to use sync invoke in Streamlit

**Solution:** Use async properly:
```python
import asyncio

async def stream():
    async for event in agent.astream_events(...):
        # Update UI
        pass

asyncio.run(stream())
```

### Issue: FastAPI endpoint timeout

**Cause:** Long-running agent blocking request

**Solution:** Add timeout:
```python
import asyncio

async def chat(request: ChatRequest):
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({...}),
            timeout=30.0  # 30 second timeout
        )
    except asyncio.TimeoutError:
        return {"error": "Request timed out"}
```

---

## General Debugging Tips

### 1. Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langgraph")
logger.setLevel(logging.DEBUG)
```

### 2. Print State at Each Step

```python
def debug_node(state):
    print(f"State: {state}")
    # Your logic
    return {...}
```

### 3. Visualize the Graph

```python
from IPython.display import Image, display

display(Image(agent.get_graph().draw_mermaid_png()))
```

### 4. Check Message History

```python
state = agent.get_state(config)
for i, msg in enumerate(state.values["messages"]):
    print(f"{i}. {type(msg).__name__}: {msg.content[:50]}...")
```

### 5. Inspect Tool Calls  

```python
for msg in result["messages"]:
    if hasattr(msg, "tool_calls"):
        print(f"Tool calls: {msg.tool_calls}")
```

---

## Performance Issues

### Slow Response Times

**Causes:**
1. Too many LLM calls (multi-agent)
2. Sequential tool execution (should be parallel)
3. Large message history (trim it)
4. No caching

**Solutions:**
```python
# 1. Cache LLM responses
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# 2. Use faster model
llm = ChatOpenAI(model="gpt-4o-mini")  # Faster than gpt-4o

# 3. Parallel tools (Module 05)
# 4. Trim history (Module 04)
```

### High Token Usage

**Monitor:**
```python
from langfuse import Langfuse

langfuse = Langfuse()
# Track token usage per request
```

**Optimize:**
- Shorter system prompts
- Trim old messages
- Use cheaper model for simple tasks

---

## Getting Help

1. **Check official docs:** https://langchain-ai.github.io/langgraph/
2. **Search GitHub issues:** https://github.com/langchain-ai/langgraph/issues
3. **Join Discord:** LangChain community server
4. **Review module code:** All working examples in `code/` directory

---

## Quick Checklist for Each Module

### Module 02
- [ ] START edge added
- [ ] add_messages reducer used
- [ ] Messages accumulating correctly

### Module 03
- [ ] Tools bound to LLM
- [ ] ToolNode in graph
- [ ] tools_condition for routing
- [ ] Tool docstrings clear

### Module 04
- [ ] Checkpointer passed to compile
- [ ] Config with thread_id in invoke
- [ ] Same thread_id for conversation
- [ ] State persisting as expected

### Module 05
- [ ] interrupt() check in code
- [ ] Max iterations for multi-agent
- [ ] Error handling in place
- [ ] Retries configured

### Module 06
- [ ] Async properly used
- [ ] Event types correct for streaming
- [ ] Rate limiting per user
- [ ] Timeouts configured

### Module 07
- [ ] All features integrated
- [ ] UI responsive
- [ ] API endpoints working
- [ ] Error handling comprehensive
