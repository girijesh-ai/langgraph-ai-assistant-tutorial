"""
Module 05: Advanced Agent Patterns
===================================
Advanced LangGraph patterns for production agents:
- Human-in-the-loop (HITL) with interrupt
- Multi-agent with supervisor pattern
- Error handling and retries
- Subgraphs for modularity
"""

import time
from datetime import datetime
from functools import wraps
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()


# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =============================================================================
# PART 1: Human-in-the-Loop (HITL)
# =============================================================================

class HITLState(TypedDict):
    """State for HITL agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    pending_action: str
    action_approved: bool


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient.
    
    Args:
        to: Email recipient
        subject: Email subject line
        body: Email body content
    
    Returns:
        Confirmation message
    """
    # In production, this would actually send an email
    return f"Email sent to {to} with subject '{subject}'"


@tool
def draft_email(to: str, subject: str, body: str) -> str:
    """Draft an email for review before sending.
    
    Args:
        to: Email recipient
        subject: Email subject line  
        body: Email body content
    
    Returns:
        Draft confirmation
    """
    return f"DRAFT - To: {to}, Subject: {subject}, Body: {body}"


def hitl_agent_node(state: HITLState) -> dict:
    """Agent node that handles HITL for sensitive actions."""
    messages = state["messages"]
    
    # Check for pending action awaiting approval
    if state.get("pending_action") and not state.get("action_approved"):
        # Ask for human approval using interrupt
        approval = interrupt({
            "action": "send_email",
            "description": "Agent wants to send an email",
            "details": state["pending_action"],
            "question": "Do you approve sending this email? (yes/no)"
        })
        
        if approval.get("approved"):
            return {
                "messages": [AIMessage(content=f"Email sent successfully! {state['pending_action']}")],
                "action_approved": True,
                "pending_action": ""
            }
        else:
            return {
                "messages": [AIMessage(content="Email sending cancelled by user.")],
                "action_approved": False,
                "pending_action": ""
            }
    
    # Normal agent processing
    response = llm.invoke(messages)
    
    # Check if agent wants to send email
    content = response.content.lower()
    if "send email" in content or "sending email" in content:
        return {
            "messages": [response],
            "pending_action": response.content,
            "action_approved": False
        }
    
    return {"messages": [response]}


def build_hitl_agent():
    """Build agent with human-in-the-loop."""
    graph = StateGraph(HITLState)
    graph.add_node("agent", hitl_agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def demo_hitl():
    """Demonstrate human-in-the-loop pattern."""
    print("=" * 70)
    print("Demo: Human-in-the-Loop (HITL)")
    print("=" * 70)
    
    agent = build_hitl_agent()
    config = {"configurable": {"thread_id": "hitl-demo"}}
    
    print("\nScenario: Agent requests to send an email, requires approval")
    print("Note: In this demo, interrupt() is simulated.\n")
    
    # First invocation
    result = agent.invoke({
        "messages": [
            SystemMessage(content="You are an email assistant. Help users send emails."),
            HumanMessage(content="Please send an email to john@example.com about the meeting tomorrow.")
        ],
        "pending_action": "",
        "action_approved": False
    }, config=config)
    
    print(f"Agent: {result['messages'][-1].content}")
    
    if result.get("pending_action"):
        print(f"\n[HITL] Action pending approval: {result['pending_action'][:50]}...")
        print("[HITL] In production, agent would pause here for user approval")


# =============================================================================
# PART 2: Multi-Agent Supervisor Pattern
# =============================================================================

class SupervisorState(TypedDict):
    """State for supervisor-based multi-agent system."""
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    task_complete: bool


def researcher_node(state: SupervisorState) -> dict:
    """Research agent - gathers information."""
    messages = state["messages"]
    
    response = llm.invoke([
        SystemMessage(content="You are a research agent. Gather relevant information about the topic."),
        *messages
    ])
    
    return {"messages": [AIMessage(content=f"[Researcher] {response.content}")]}


def writer_node(state: SupervisorState) -> dict:
    """Writer agent - creates content."""
    messages = state["messages"]
    
    response = llm.invoke([
        SystemMessage(content="You are a writer agent. Create well-written content based on research."),
        *messages
    ])
    
    return {"messages": [AIMessage(content=f"[Writer] {response.content}")]}


def reviewer_node(state: SupervisorState) -> dict:
    """Reviewer agent - checks quality."""
    messages = state["messages"]
    
    response = llm.invoke([
        SystemMessage(content="You are a reviewer agent. Review the content for quality and accuracy."),
        *messages
    ])
    
    return {"messages": [AIMessage(content=f"[Reviewer] {response.content}")]}


def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor - coordinates other agents."""
    messages = state["messages"]
    
    response = llm.invoke([
        SystemMessage(content="""You are a supervisor coordinating a research, writing, and review team.
        Based on the current state, decide the next step:
        - 'researcher': Need more information
        - 'writer': Ready to write content
        - 'reviewer': Content needs review
        - 'FINISH': Task is complete
        
        Respond with just the agent name or FINISH."""),
        *messages
    ])
    
    next_agent = response.content.strip().lower()
    
    return {
        "next_agent": next_agent,
        "task_complete": next_agent == "finish"
    }


def route_supervisor(state: SupervisorState) -> Literal["researcher", "writer", "reviewer", "__end__"]:
    """Route to the appropriate agent based on supervisor decision."""
    next_agent = state.get("next_agent", "").lower()
    
    if next_agent == "finish" or state.get("task_complete"):
        return "__end__"
    elif next_agent == "researcher":
        return "researcher"
    elif next_agent == "writer":
        return "writer"
    elif next_agent == "reviewer":
        return "reviewer"
    else:
        return "__end__"


def build_supervisor_agent():
    """Build multi-agent system with supervisor."""
    graph = StateGraph(SupervisorState)
    
    # Add all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    
    # Start with supervisor
    graph.add_edge(START, "supervisor")
    
    # Supervisor routes to agents
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "__end__": END
        }
    )
    
    # All agents return to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")
    
    return graph.compile()


def demo_supervisor():
    """Demonstrate supervisor pattern."""
    print("=" * 70)
    print("Demo: Multi-Agent Supervisor Pattern")
    print("=" * 70)
    
    agent = build_supervisor_agent()
    
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Write a brief article about the benefits of AI agents in business.")
        ],
        "next_agent": "",
        "task_complete": False
    })
    
    print("\nConversation flow:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n[Task completed: {result.get('task_complete', False)}]")


# =============================================================================
# PART 3: Error Handling and Retries
# =============================================================================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    last_error = e
                    
                    if retries >= max_retries:
                        break
                        
                    delay = base_delay * (2 ** retries)
                    print(f"  [Retry {retries}/{max_retries}] Error: {e}. Waiting {delay}s...")
                    time.sleep(delay)
            
            raise last_error
        return wrapper
    return decorator


@retry_with_backoff(max_retries=3, base_delay=0.5)
def unreliable_api_call(succeed_on_attempt: int, current_attempt: list) -> str:
    """Simulated unreliable API that fails initially then succeeds."""
    current_attempt[0] += 1
    
    if current_attempt[0] < succeed_on_attempt:
        raise ConnectionError("API temporarily unavailable")
    
    return "API call successful"


def demo_retry():
    """Demonstrate retry pattern."""
    print("=" * 70)
    print("Demo: Error Handling with Retry")
    print("=" * 70)
    
    print("\nScenario: API fails twice, succeeds on third attempt")
    attempt_counter = [0]  # Using list to allow modification in closure
    
    try:
        result = unreliable_api_call(succeed_on_attempt=3, current_attempt=attempt_counter)
        print(f"\n[Success] {result} after {attempt_counter[0]} attempts")
    except Exception as e:
        print(f"\n[Failed] {e} after {attempt_counter[0]} attempts")


class ErrorHandlingState(TypedDict):
    """State for error-handling demo."""
    messages: Annotated[list[BaseMessage], add_messages]
    error_count: int


def error_handling_node(state: ErrorHandlingState) -> dict:
    """Node with comprehensive error handling."""
    try:
        # Simulate processing
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response], "error_count": 0}
        
    except Exception as e:
        error_count = state.get("error_count", 0) + 1
        
        if error_count >= 3:
            return {
                "messages": [AIMessage(
                    content="I apologize, but I am experiencing technical difficulties. "
                            "Please try again later or contact support."
                )],
                "error_count": error_count
            }
        
        return {
            "messages": [AIMessage(
                content=f"An error occurred. Attempting recovery... (attempt {error_count}/3)"
            )],
            "error_count": error_count
        }


# =============================================================================
# PART 4: Subgraphs
# =============================================================================

class SubgraphState(TypedDict):
    """State for subgraph demo."""
    messages: Annotated[list[BaseMessage], add_messages]
    research_complete: bool
    summary: str


def search_node(state: SubgraphState) -> dict:
    """Search for information."""
    messages = state["messages"]
    query = messages[-1].content if messages else "general topic"
    
    return {
        "messages": [AIMessage(content=f"[Search] Found information about: {query}")],
        "research_complete": False
    }


def summarize_node(state: SubgraphState) -> dict:
    """Summarize findings."""
    messages = state["messages"]
    
    response = llm.invoke([
        SystemMessage(content="Summarize the research findings briefly."),
        *messages
    ])
    
    return {
        "messages": [AIMessage(content=f"[Summary] {response.content}")],
        "research_complete": True,
        "summary": response.content
    }


def build_research_subgraph():
    """Build a reusable research subgraph."""
    graph = StateGraph(SubgraphState)
    
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    
    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)
    
    return graph.compile()


def main_agent_node(state: SubgraphState) -> dict:
    """Main agent that uses research results."""
    summary = state.get("summary", "No research available")
    
    response = llm.invoke([
        SystemMessage(content="Use the research summary to provide a helpful response."),
        HumanMessage(content=f"Research summary: {summary}"),
        *state["messages"]
    ])
    
    return {"messages": [response]}


def demo_subgraph():
    """Demonstrate subgraph composition."""
    print("=" * 70)
    print("Demo: Subgraph Composition")
    print("=" * 70)
    
    # Build subgraph
    research_subgraph = build_research_subgraph()
    
    # Use subgraph directly
    result = research_subgraph.invoke({
        "messages": [HumanMessage(content="Tell me about LangGraph")],
        "research_complete": False,
        "summary": ""
    })
    
    print("\nSubgraph execution:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"  {msg.content[:100]}...")
    
    print(f"\nResearch complete: {result.get('research_complete')}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run all advanced pattern demonstrations."""
    # Demo 1: Human-in-the-loop
    demo_hitl()
    print()
    
    # Demo 2: Supervisor pattern
    demo_supervisor()
    print()
    
    # Demo 3: Error handling
    demo_retry()
    print()
    
    # Demo 4: Subgraphs
    demo_subgraph()


if __name__ == "__main__":
    main()
