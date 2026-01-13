"""
Module 07: Streamlit App
=========================
A beautiful chat interface for the AI assistant using Streamlit.

Run with: streamlit run 07_streamlit_app.py
"""

import asyncio
import os
import sys
from datetime import datetime

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict

# Load environment
load_dotenv()

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="AI Assistant",
    page_icon="robot",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =============================================================================
# State and Tools (same as complete_agent)
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Only basic math operations allowed"
    try:
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
def search_knowledge(query: str) -> str:
    """Search the knowledge base for information."""
    kb = {
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "tools": "Tools allow LLMs to take actions and interact with external systems.",
        "memory": "LangGraph supports memory through checkpointers for persistent conversations.",
        "streaming": "Streaming delivers tokens in real-time as the LLM generates them.",
    }
    
    for key, value in kb.items():
        if key in query.lower():
            return value
    return f"No specific information found for: {query}"


tools = [calculator, get_current_time, search_knowledge]


# =============================================================================
# Agent Build
# =============================================================================

@st.cache_resource
def build_agent():
    """Build and cache the agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    llm_with_tools = llm.bind_tools(tools)
    
    async def agent_node(state):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}
    
    tool_node = ToolNode(tools)
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


agent = build_agent()


# =============================================================================
# Session State
# =============================================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session-{datetime.now().timestamp()}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False


def init_conversation():
    """Initialize conversation with system message."""
    if not st.session_state.initialized:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        asyncio.run(agent.ainvoke({
            "messages": [SystemMessage(
                content="You are a helpful AI assistant. Use tools when needed. Be concise."
            )]
        }, config=config))
        st.session_state.initialized = True


init_conversation()


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("Settings")
    
    st.markdown("---")
    
    st.subheader("Available Tools")
    st.markdown("""
    - **calculator**: Math operations
    - **get_current_time**: Current datetime
    - **search_knowledge**: Knowledge base
    """)
    
    st.markdown("---")
    
    if st.button("New Conversation"):
        st.session_state.thread_id = f"session-{datetime.now().timestamp()}"
        st.session_state.messages = []
        st.session_state.initialized = False
        st.rerun()
    
    st.markdown("---")
    
    st.caption(f"Thread: {st.session_state.thread_id[:15]}...")
    st.caption(f"Messages: {len(st.session_state.messages)}")


# =============================================================================
# Main Chat Interface
# =============================================================================

st.title("AI Assistant")
st.caption("A LangGraph-powered assistant with tools and memory")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Validate input
    if len(prompt) > 4000:
        st.error("Message too long. Maximum 4000 characters.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            tool_info = []
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Collect response
            async def stream():
                nonlocal full_response, tool_info
                
                async for event in agent.astream_events(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    version="v2"
                ):
                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            full_response += chunk.content
                            message_placeholder.markdown(full_response + " |")
                    
                    elif event["event"] == "on_tool_start":
                        tool_info.append(f"Using: {event['name']}")
                    
                    elif event["event"] == "on_tool_end":
                        output = str(event["data"].get("output", ""))[:50]
                        tool_info.append(f"Result: {output}...")
                
                message_placeholder.markdown(full_response)
            
            asyncio.run(stream())
            
            # Show tool usage if any
            if tool_info:
                with st.expander("Tool Usage"):
                    for info in tool_info:
                        st.text(info)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        Built with LangGraph | Module 07 Complete Agent
    </div>
    """,
    unsafe_allow_html=True
)
