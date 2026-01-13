"""
Module 02: Basic Chatbot Foundation
=====================================
A simple LangGraph chatbot demonstrating core concepts:
- State management with MessagesState pattern
- Single chat node with LLM
- Graph compilation and execution
- Multi-turn conversation handling
"""

import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()


# -----------------------------------------------------------------------------
# State Definition
# -----------------------------------------------------------------------------
class ChatState(TypedDict):
    """State for a simple chatbot.
    
    The messages field uses add_messages annotation which:
    - Appends new messages to existing list
    - Handles message deduplication by ID
    - Supports message updates/deletions
    """
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
def get_llm():
    """Create and return the LLM instance.
    
    Uses OpenAI by default. For local LLMs, uncomment the Ollama section.
    """
    # OpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    
    # Uncomment for Ollama (local)
    # from langchain_ollama import ChatOllama
    # llm = ChatOllama(
    #     model="llama3.1",
    #     base_url="http://localhost:11434",
    #     temperature=0.7,
    # )
    
    return llm


# Initialize LLM
llm = get_llm()


# -----------------------------------------------------------------------------
# Graph Nodes
# -----------------------------------------------------------------------------
def chat_node(state: ChatState) -> dict:
    """Process messages and generate a response.
    
    Args:
        state: Current graph state with message history
        
    Returns:
        Dictionary with new messages to add to state
    """
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# -----------------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------------
def build_chatbot():
    """Build and compile the chatbot graph."""
    # Create graph with state schema
    graph = StateGraph(ChatState)
    
    # Add the chat node
    graph.add_node("chat", chat_node)
    
    # Define edges
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    
    # Compile and return
    return graph.compile()


# Create the chatbot
chatbot = build_chatbot()


# -----------------------------------------------------------------------------
# Conversation Handlers
# -----------------------------------------------------------------------------
def single_turn_example():
    """Demonstrate a single-turn conversation."""
    print("=" * 60)
    print("Single-Turn Example")
    print("=" * 60)
    
    result = chatbot.invoke({
        "messages": [HumanMessage(content="What is the capital of Japan?")]
    })
    
    print(f"User: What is the capital of Japan?")
    print(f"Assistant: {result['messages'][-1].content}")
    print()


def multi_turn_example():
    """Demonstrate manual multi-turn conversation."""
    print("=" * 60)
    print("Multi-Turn Example (Manual History)")
    print("=" * 60)
    
    # Turn 1
    messages = [
        SystemMessage(content="You are a helpful assistant. Be concise."),
        HumanMessage(content="My name is Alice and I love Python.")
    ]
    result = chatbot.invoke({"messages": messages})
    print(f"User: My name is Alice and I love Python.")
    print(f"Assistant: {result['messages'][-1].content}")
    print()
    
    # Turn 2 - Include previous history
    messages = result["messages"] + [
        HumanMessage(content="What is my name and what do I love?")
    ]
    result = chatbot.invoke({"messages": messages})
    print(f"User: What is my name and what do I love?")
    print(f"Assistant: {result['messages'][-1].content}")
    print()


async def streaming_example():
    """Demonstrate streaming responses."""
    print("=" * 60)
    print("Streaming Example")
    print("=" * 60)
    
    print("User: Explain recursion in one paragraph.")
    print("Assistant: ", end="", flush=True)
    
    async for event in chatbot.astream_events(
        {"messages": [HumanMessage(content="Explain recursion in one paragraph.")]},
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
    
    print("\n")


def interactive_chat():
    """Run an interactive chat session."""
    print("=" * 60)
    print("Interactive Chat")
    print("Type 'quit', 'exit', or 'q' to end.")
    print("=" * 60)
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant. Be concise and friendly.")
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
        
        # Add user message
        messages.append(HumanMessage(content=user_input))
        
        # Get response
        result = chatbot.invoke({"messages": messages})
        
        # Update messages with full history
        messages = result["messages"]
        
        # Print response
        print(f"Assistant: {messages[-1].content}")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    """Run all examples."""
    # Single turn
    single_turn_example()
    
    # Multi-turn
    multi_turn_example()
    
    # Streaming (async)
    asyncio.run(streaming_example())
    
    # Interactive (optional - uncomment to use)
    # interactive_chat()


if __name__ == "__main__":
    main()
