# System Architecture

This document provides a detailed architecture overview of the complete AI assistant built in Module 07.

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Interfaces"
        UI1[Streamlit UI]
        UI2[FastAPI Server]
        UI3[CLI Interface]
    end
    
    subgraph "Application Layer"
        RL[Rate Limiter]
        IV[Input Validator]
        SH[Streaming Handler]
    end
    
    subgraph "LangGraph Agent Core"
        direction TB
        START([START])
        AGENT[Agent Node<br/>LLM Reasoning]
        TOOLS[Tool Node<br/>Action Execution]
        END([END])
        
        START --> AGENT
        AGENT -->|tool_calls?| DECISION{tools_condition}
        DECISION -->|Yes| TOOLS
        DECISION -->|No| END
        TOOLS --> AGENT
    end
    
    subgraph "Tools Layer"
        T1[Calculator]
        T2[Time/Date]
        T3[Knowledge Search]
        T4[String Operations]
    end
    
    subgraph "Memory and State"
        CP[Checkpointer<br/>MemorySaver/SqliteSaver]
        STATE[(State Storage<br/>messages, metadata)]
        THREAD[Thread Management<br/>session IDs]
    end
    
    subgraph "LLM Provider"
        LLM[OpenAI GPT-4o-mini<br/>with tools bound]
    end
    
    subgraph "Observability"
        LOG[Structured Logging]
        TRACE[Langfuse/LangSmith<br/>Tracing]
    end
    
    UI1 --> RL
    UI2 --> RL
    UI3 --> RL
    
    RL --> IV
    IV --> SH
    SH --> AGENT
    
    AGENT <--> LLM
    TOOLS --> T1
    TOOLS --> T2
    TOOLS --> T3
    TOOLS --> T4
    
    AGENT <--> CP
    CP <--> STATE
    CP <--> THREAD
    
    AGENT -.-> LOG
    AGENT -.-> TRACE
    
    style AGENT fill:#e1f5ff
    style TOOLS fill:#fff4e1
    style CP fill:#f0f4ff
    style LLM fill:#ffe1f5
```

## Detailed Component Breakdown

### 1. User Interfaces Layer

```mermaid
graph LR
    subgraph "Streamlit UI"
        ST1[Chat Input]
        ST2[Message Display]
        ST3[Session State]
        ST4[Tool Visibility]
    end
    
    subgraph "FastAPI Server"
        API1[POST /chat]
        API2[POST /chat/stream]
        API3[GET /history]
        API4[GET /health]
    end
    
    subgraph "CLI"
        CLI1[Interactive Loop]
        CLI2[Command Parser]
        CLI3[Output Formatter]
    end
    
    ST1 --> CORE[Agent Core]
    API1 --> CORE
    CLI1 --> CORE
```

### 2. Agent Graph Flow

```mermaid
stateDiagram-v2
    [*] --> Validate
    Validate --> RateLimit
    RateLimit --> AgentNode
    
    state AgentNode {
        [*] --> LoadHistory
        LoadHistory --> InvokeLLM
        InvokeLLM --> CheckToolCalls
        CheckToolCalls --> [*]
    }
    
    AgentNode --> ToolsCondition
    
    state ToolsCondition <<choice>>
    ToolsCondition --> ToolNode: tool_calls present
    ToolsCondition --> Response: no tools needed
    
    state ToolNode {
        [*] --> ExecuteTools
        ExecuteTools --> CollectResults
        CollectResults --> [*]
    }
    
    ToolNode --> AgentNode: loop back with results
    Response --> SaveCheckpoint
    SaveCheckpoint --> [*]
```

### 3. State Management

```mermaid
graph TB
    subgraph "State Structure"
        STATE[AgentState]
        MSG[messages: list]
        META[metadata: dict]
        
        STATE --> MSG
        STATE --> META
    end
    
    subgraph "Message Types"
        HUMAN[HumanMessage]
        AI[AIMessage]
        TOOL[ToolMessage]
        SYS[SystemMessage]
    end
    
    subgraph "Checkpointer"
        MEM[MemorySaver<br/>Development]
        SQL[SqliteSaver<br/>Production]
    end
    
    MSG --> HUMAN
    MSG --> AI
    MSG --> TOOL
    MSG --> SYS
    
    STATE --> MEM
    STATE --> SQL
    
    MEM --> THREAD1[Thread: user-123]
    SQL --> THREAD2[Thread: user-456]
```

### 4. Tool Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant ToolNode
    participant Tool
    
    User->>Agent: "What is 25 * 4?"
    Agent->>LLM: Invoke with message history
    LLM->>Agent: AIMessage with tool_calls
    Agent->>ToolNode: Execute tool_calls
    ToolNode->>Tool: calculator("25 * 4")
    Tool->>ToolNode: "Result: 100"
    ToolNode->>Agent: ToolMessage with result
    Agent->>LLM: Invoke with updated history
    LLM->>Agent: AIMessage with final answer
    Agent->>User: "The result is 100"
```

### 5. Production Features Stack

```mermaid
graph TD
    REQ[User Request] --> V1[Input Validation]
    V1 --> V2[Rate Limiter]
    V2 --> V3[Timeout Wrapper]
    V3 --> AGENT[Agent Core]
    
    AGENT --> S1[Streaming Handler]
    S1 --> S2[Token Aggregator]
    S2 --> RESP[Response]
    
    AGENT -.-> L1[Structured Logger]
    AGENT -.-> L2[Langfuse Tracer]
    AGENT -.-> L3[Error Handler]
    
    style V1 fill:#ffe1e1
    style V2 fill:#ffe1e1
    style V3 fill:#ffe1e1
    style L1 fill:#e1ffe1
    style L2 fill:#e1ffe1
    style L3 fill:#e1ffe1
```

### 6. Data Flow

```mermaid
flowchart LR
    subgraph Input
        I1[User Message]
        I2[Thread ID]
        I3[Config]
    end
    
    subgraph Processing
        P1[Load Checkpoint]
        P2[Add Message to State]
        P3[Execute Graph]
        P4[Save Checkpoint]
    end
    
    subgraph Output
        O1[AI Response]
        O2[Tool Calls Used]
        O3[Updated State]
    end
    
    I1 --> P1
    I2 --> P1
    I3 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> O1
    P4 --> O2
    P4 --> O3
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Graph Framework** | LangGraph |
| **LLM** | OpenAI GPT-4o-mini |
| **State Management** | MemorySaver / SqliteSaver |
| **Tools** | LangChain Tools (@tool decorator) |
| **Streaming** | astream_events (v2) |
| **UI** | Streamlit |
| **API** | FastAPI + Uvicorn |
| **Observability** | Langfuse / LangSmith |
| **Environment** | Python 3.10+, python-dotenv |

## Key Patterns Implemented

1. **ReAct Pattern**: Reasoning (LLM) + Acting (Tools) loop
2. **Checkpointing**: Persistent conversation state across sessions
3. **Streaming**: Real-time token delivery via async events
4. **HITL**: Human-in-the-loop with `interrupt()`
5. **Multi-threading**: Separate conversations via thread_id
6. **Rate Limiting**: Request throttling per user
7. **Input Validation**: Security against injection attacks
8. **Error Handling**: Graceful degradation with retries

## Scalability Considerations

```mermaid
graph TB
    subgraph "Development"
        D1[In-Memory State]
        D2[Local LLM]
        D3[Single Instance]
    end
    
    subgraph "Production"
        P1[PostgreSQL Checkpointer]
        P2[OpenAI API]
        P3[Load Balancer]
        P4[Redis Cache]
        P5[Message Queue]
    end
    
    D1 -.Upgrade.-> P1
    D2 -.Upgrade.-> P2
    D3 -.Upgrade.-> P3
    D3 -.Add.-> P4
    D3 -.Add.-> P5
```

---

For implementation details, see [Module 07: Complete Agent](07_complete_agent.md)
