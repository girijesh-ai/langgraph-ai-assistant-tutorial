# Module 02: Building the Basic Chatbot Foundation

In this module, you will build a foundational chatbot using LangGraph. This forms the base upon which we will add tools, memory, and advanced features in subsequent modules.

## Learning Objectives

By the end of this module, you will:
- Understand the MessagesState pattern for chat applications
- Create a simple chat node that processes messages
- Compile and run a basic LangGraph workflow
- Handle multi-turn conversations

## The MessagesState Pattern

LangGraph provides a built-in `MessagesState` that handles the common pattern of accumulating messages in a conversation. This uses the `add_messages` reducer to automatically append new messages to the existing list.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """State for a simple chatbot.
    
    The messages field uses add_messages annotation which:
    - Appends new messages to existing list
    - Handles message deduplication by ID
    - Supports message updates/deletions
    """
    messages: Annotated[list[BaseMessage], add_messages]
```

### Why Use add_messages?

The `add_messages` reducer is crucial for chat applications:

1. **Accumulation**: New messages are appended, maintaining conversation history
2. **Deduplication**: Messages with same ID are updated, not duplicated
3. **Deletion**: Special `RemoveMessage` objects can remove messages
4. **Type Safety**: Works with all LangChain message types

## Building the Chat Node

A node in LangGraph is simply a function that takes state and returns updated state:

```python
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def chat_node(state: ChatState) -> dict:
    """Process messages and generate a response.
    
    Args:
        state: Current graph state with message history
        
    Returns:
        Dictionary with new messages to add to state
    """
    # Get all messages from state
    messages = state["messages"]
    
    # Send to LLM and get response
    response = llm.invoke(messages)
    
    # Return new message(s) to be added to state
    return {"messages": [response]}
```

### Key Points

1. **Input**: The node receives the full state object
2. **Processing**: Extract messages, invoke LLM
3. **Output**: Return a dict with keys matching state fields
4. **Reducer**: The `add_messages` reducer handles appending

## Assembling the Graph

Now we connect everything together:

```python
from langgraph.graph import StateGraph, START, END

# Create the graph with our state schema
graph = StateGraph(ChatState)

# Add the chat node
graph.add_node("chat", chat_node)

# Connect: START -> chat -> END
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# Compile into a runnable
chatbot = graph.compile()
```

### Graph Flow Visualization

```
    START
      |
      v
  +-------+
  | chat  |  <-- LLM processes messages
  +-------+
      |
      v
     END
```

## Running the Chatbot

### Single Invocation

```python
from langchain_core.messages import HumanMessage

# First message
result = chatbot.invoke({
    "messages": [HumanMessage(content="What is the capital of France?")]
})

print(result["messages"][-1].content)
# Output: The capital of France is Paris...
```

### Multi-turn Conversation (Manual)

Without memory/checkpointing, you must manually maintain history:

```python
# First turn
messages = [HumanMessage(content="My name is Alice")]
result = chatbot.invoke({"messages": messages})

# Second turn - include previous messages
messages = result["messages"] + [HumanMessage(content="What is my name?")]
result = chatbot.invoke({"messages": messages})

print(result["messages"][-1].content)
# Output: Your name is Alice...
```

## Adding Basic Streaming

For a better user experience, stream tokens as they are generated:

```python
async def stream_chat(user_input: str, messages: list):
    """Stream response tokens."""
    messages = messages + [HumanMessage(content=user_input)]
    
    async for event in chatbot.astream_events(
        {"messages": messages},
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
    print()  # Newline after response
```

## Complete Code

See [code/02_basic_chatbot.py](code/02_basic_chatbot.py) for the complete working example.

## Common Issues

### Issue: LLM Not Responding
- Check your API key is set correctly
- Verify network connectivity
- Check rate limits on your API account

### Issue: Messages Not Accumulating
- Ensure you are using `add_messages` annotation
- Check that node returns `{"messages": [response]}`

## Exercises

1. **Temperature Experiment**: Modify the temperature parameter and observe response variation
2. **System Message**: Add a system message to give the bot a personality
3. **Message Logging**: Add logging to see full message history at each turn

## Next Steps

In [Module 03](03_tools.md), we will add tool calling capabilities so our agent can take actions - calculate, search, and more.

---

[Back to README](README.md) | [Next: Tools and Actions](03_tools.md)
