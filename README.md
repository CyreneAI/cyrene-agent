````markdown
# 🧠 Cyrene Agent (bot-api)

This repository contains the core bot-api service of the Multi-Agent Bot system. It acts as the central brain, responsible for managing AI agents, orchestrating their interactions with Large Language Models (LLMs) and various tools, and handling communication with external platforms via specialized MCP (Multi-Channel Platform) servers.

## ✨ Features

- **Agent Lifecycle Management**: Create, list, retrieve, and manage AI agent configurations (persona, bio, knowledge, secrets).
- **Intelligent Agent Orchestration**: Utilizes LangChain and LangGraph to power agents capable of advanced reasoning, planning, and dynamic tool selection.
- **LLM Integration**: Seamlessly integrates with Groq LLMs for fast and efficient language processing.
- **Dynamic Tool Invocation**: Connects with the fastmcp-core-server to dynamically discover and invoke specialized tools (web search, finance, RAG, platform-specific actions).
- **Platform Message Handling**: Receives standardized messages from platform-specific MCPs (Discord, Telegram), routes them to the correct agent, and facilitates agent replies.
- **SQLite for Persistence**: Stores agent configurations in a local SQLite database.
- **Scalable API**: Built with FastAPI for high performance and easy extensibility.

## 🏛️ Architecture Context

The cyrene-agent (bot-api) is the core backend service. It exposes RESTful endpoints for the frontend (agent-UI) to manage agents and for platform MCPs to send incoming messages. It then leverages MultiServerMCPClient to interact with all other specialized MCP services (e.g., web-mcp, finance-mcp, rag-mcp, telegram-mcp, discord-mcp) to execute tools.

```mermaid
graph TD
    Agent_UI -->|HTTP API| agent_api
    Platform_MCPs -->|HTTP Webhook/API| agent_api
    agent_api -->|HTTP API| Specialized_MCPs
    agent_api -- SQLite --> Internal_DB[SQLite Database]

    subgraph This Repository (cyrene-agent)
        agent_api
        agent_core[agent/core/agent_manager.py]
        agent_db[cyrene-agent/db/sqlite_manager.py]
        langgraph_agents[agent/langgraph_agents/custom_tool_agent.py]
        models[agent/models/agent_config.py]
        prompts[prompts.py]
        cyrene-agent --> agent_core
        cyrene-agent --> agent_db
        cyrene-agent --> langgraph_agents
        cyrene-agent --> models
        cyrene-agent --> prompts
    end

    subgraph External Services
        Platform_MCPs[Telegram/Discord MCPs]
        Specialized_MCPs[Web/Finance/RAG MCPs]
    end
````

## 🚀 Getting Started

### Prerequisites

* Python 3.12+
* Groq/OpenAI API Key
* Running MCP Servers: This service relies on the fastmcp-core-server and other specialized MCPs (web, finance, RAG, Telegram, Discord) being accessible.

### Installation

Clone this repository:

```bash
git clone https://github.com/CyreneAI/cyrene-agent.git
cd cyrene-agent
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root of this `cyrene-agent` directory (bot/ directory for the Dockerfile context) with the following variables:

```env
# .env in cyrene-agent/bot directory
GROQ_API_KEY=sk_your_groq_api_key
LOCAL_MODE=true
SQLITE_DB_PATH=agents.db
```

* `GROQ_API_KEY`: Your API key for Groq.
* `LOCAL_MODE`: Set to `true` for local development (MCPs on localhost:900x), `false` for Kubernetes deployment.
* `SQLITE_DB_PATH`: Path to your SQLite database file (e.g., `agents.db`).

### Running the Application (Local Development)

Ensure all necessary MCP servers are running (Web, Finance, RAG, Telegram, Discord).

Run the bot-api service:

```bash
uvicorn bot.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be accessible at `http://localhost:8000`.

## 🧪 Usage

* **Agent Creation**: Use the `/agents/create` endpoint (typically via the agent-UI frontend) to register new agents. Provide `discord_bot_token` or `telegram_bot_token`/`api_id`/`api_hash` in the `secrets` field if the agent needs platform capabilities.
* **Chat with Agents**: Use the `/agents/{agent_id}/chat` endpoint (via frontend) to send messages to an agent.
* **Receive Platform Messages**: The `/telegram/webhook` and `/discord/receive_message` endpoints are designed to receive messages forwarded from the respective platform MCPs.

## 📁 Project Structure

```
cyrene-agent/
├── .env.example
├── .gitignore
├── README.md           # <- This file
│── Dockerfile          # Dockerfile for the bot-api service
├── __init__.py
├── api/
│   ├── __init__.py
│   └── main.py         # FastAPI application for bot-api
├── core/
│   ├── __init__.py
│   └── agent_manager.py # Manages agent lifecycle and tool wrapping
├── db/
│   ├── __init__.py
│   └── sqlite_manager.py # Handles SQLite database operations
├── langgraph_agents/
│   └── custom_tool_agent.py # Defines the LangGraph agent structure
├── models/
│   ├── __init__.py
│   └── agent_config.py  # Pydantic models for AgentConfig and AgentSecrets
├── prompts.py          # LLM system prompts
└── requirements.txt    # Python dependencies for agent-api

```
