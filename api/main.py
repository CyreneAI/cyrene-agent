import logging
import json 
import uuid 
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from models.agent_config import AgentConfig, AgentSecrets, Settings 
from core.agent_manager import AgentManager 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


# --------- Load environment variables ---------
load_dotenv()

# --------- Logging Setup ---------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- Pydantic Model for Discord Message Payload ---------
class ReceiveDiscordMessageRequest(BaseModel):
    content: str
    channel_id: str
    author_id: str
    author_name: str
    message_id: str
    timestamp: str
    guild_id: Optional[str] = None
    bot_id: str

# --------- Determine Local or Cluster Mode ---------
LOCAL_MODE = "false"
logger.info(f"Running in LOCAL_MODE: {LOCAL_MODE}")

# Initialize AgentManager instance globally
DB_PATH = "/app/data/agents.db"
agent_manager_instance = AgentManager(DB_PATH)


# --------- FastAPI Lifespan Context Manager ---------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for initializing and cleaning up resources.
    Initializes the SQLite database and agents.
    """
    logger.info("Agent app startup: Initializing global resources...")
    
    app.state.db_manager = agent_manager_instance.db_manager 
    logger.info(f"SQLite database '{DB_PATH}' initialized by SQLiteManager.")

    try:
        await agent_manager_instance.initialize_all_agents_from_db(LOCAL_MODE)
    except Exception as e:
        logger.error(f"Failed to initialize agents on startup: {e}", exc_info=True)

    logger.info("Agent app startup complete. Agent is ready.")
    yield
    
    logger.info("Agent app shutdown.")
    await agent_manager_instance.shutdown_all_agents()
    app.state.db_manager.close() 
    logger.info("SQLite database connection closed.")

app = FastAPI(lifespan=lifespan, debug=True)

# --------- FastAPI Endpoints ---------

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Multi-Agent Bot API!"}

@app.post("/agents/create", response_model=AgentConfig, status_code=status.HTTP_201_CREATED)
async def create_agent(agent_config_data: Dict[str, Any]): 
    try:
        settings_data = agent_config_data.get("settings", {})
        secrets_from_json = settings_data.get("secrets", {})
        voice_settings = settings_data.get("voice", {})

        # --- SIMPLIFIED SECRET PARSING ---
        # Directly pass the secrets_from_json dict to AgentSecrets Pydantic model
        # Pydantic will handle validation and mapping to the correct fields.
        agent_secrets_instance = AgentSecrets(**secrets_from_json)
        logger.debug(f"[main.py] Parsed AgentSecrets: {agent_secrets_instance.model_dump_json(exclude_none=True)}")

        # Create Settings instance
        settings_instance = Settings(
            model=settings_data.get("model", "llama3-8b-8192"),
            temperature=settings_data.get("temperature", 0.7),
            maxTokens=settings_data.get("maxTokens", 8192),
            secrets=agent_secrets_instance,
            voice=voice_settings if voice_settings else None
        )

        # Create AgentConfig object
        agent_config = AgentConfig(
            id=str(uuid.uuid4()), 
            name=agent_config_data.get("name", "NewBot"),
            modelProvider=agent_config_data.get("modelProvider", "groq"), 
            settings=settings_instance, 
            system=agent_config_data.get("system", ""), # Frontend sends 'system'
            bio=agent_config_data.get("bio", []), 
            lore=agent_config_data.get("lore", []), 
            knowledge=agent_config_data.get("knowledge", []), 
            messageExamples=agent_config_data.get("messageExamples"),
            style=agent_config_data.get("style")
        )

        # Save config to DB first
        agent_id = await app.state.db_manager.save_agent_config(agent_config)
        agent_config.id = agent_id

        # Dynamically create and initialize the agent instance and add it to the manager's cache
        executor, mcp_client, discord_bot_id, telegram_bot_id = \
            await agent_manager_instance.create_dynamic_agent_instance(agent_config, LOCAL_MODE)
        
        agent_manager_instance.add_initialized_agent(
            agent_config.id,
            agent_config.name,
            executor,
            mcp_client,
            discord_bot_id=discord_bot_id,
            telegram_bot_id=telegram_bot_id
        )
        logger.info(f"Agent '{agent_config.name}' (ID: {agent_config.id}) created and initialized.")
        
        return agent_config
    except ValidationError as e:
        logger.error(f"Validation Error creating agent: {e.errors()}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid agent configuration: {e.errors()}")
    except Exception as e:
        logger.error(f"Error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create agent: {e}")

@app.get("/agents/list", response_model=List[AgentConfig])
async def list_agents():
    try:
        configs = await app.state.db_manager.get_all_agent_configs()
        return configs
    except Exception as e:
        logger.error(f"Error listing agents: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list agents: {e}")

@app.delete("/agents/{agent_id}", status_code=status.HTTP_200_OK)
async def delete_agent(agent_id: str):
    """
    Endpoint for deleting an agent.
    """
    agent_info = agent_manager_instance.get_initialized_agent(agent_id)
    if not agent_info:
        agent_config_from_db = await app.state.db_manager.get_agent_config(agent_id)
        if not agent_config_from_db:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent with ID {agent_id} not found.")

    try:
        await agent_manager_instance.shutdown_specific_agent(agent_id)
        await app.state.db_manager.delete_agent_config(agent_id)

        return {"message": f"Agent '{agent_id}' deleted successfully."}
    except Exception as e:
        logger.error(f"Failed to delete agent '{agent_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete agent: {e}")


# --------- Streamlit Chat (Frontend) ---------

@app.post("/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: str, message: Dict[str, str]):
    user_message = message.get("message")
    if not user_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message content is required.")

    agent_info = agent_manager_instance.get_initialized_agent(agent_id)
    
    if not agent_info:
        agent_config = await app.state.db_manager.get_agent_config(agent_id)
        if not agent_config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent with ID '{agent_id}' not found.")
        
        # Attempt to re-initialize and add to cache
        try:
            executor, mcp_client, discord_bot_id, telegram_bot_id = \
                await agent_manager_instance.create_dynamic_agent_instance(agent_config, LOCAL_MODE)
            
            # ADDED: Add the newly created agent to the manager's cache
            agent_manager_instance.add_initialized_agent(
                agent_config.id,
                agent_config.name,
                executor,
                mcp_client,
                discord_bot_id=discord_bot_id,
                telegram_bot_id=telegram_bot_id
            )
            agent_info = agent_manager_instance.get_initialized_agent(agent_id) # Now it should be found
            logger.info(f"Agent '{agent_config.name}' (ID: {agent_config.id}) re-initialized and added to cache for chat.")
        except Exception as e:
            logger.error(f"Failed to re-initialize agent '{agent_id}' during chat request: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to initialize agent '{agent_id}' for chat: {e}")

        if not agent_info: # This check might still be needed if add_initialized_agent itself fails somehow
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve agent '{agent_id}' from cache after re-initialization attempt.")

    agent_executor = agent_info["executor"] 

    logger.info(f"Invoking agent '{agent_id}' with message: {user_message}")
    logger.info(f"DEBUG: User message being passed to agent.ainvoke: '{user_message}'")

    try:
        initial_state = {"messages": [HumanMessage(content=user_message)]}
        agent_output = await agent_executor.ainvoke(initial_state)

        final_message_content = "I'm sorry, I couldn't process that."
        if "messages" in agent_output and agent_output["messages"]:
            last_message = agent_output["messages"][-1]
            if isinstance(last_message, AIMessage):
                final_message_content = last_message.content
                if last_message.tool_calls:
                    logger.info(f"DEBUG: Agent '{agent_id}' generated tool calls (unexpected at END): {last_message.tool_calls}")
            elif isinstance(last_message, HumanMessage):
                final_message_content = last_message.content
            elif isinstance(last_message, SystemMessage):
                final_message_content = last_message.content
            elif isinstance(last_message, ToolMessage):
                final_message_content = "I processed information using a tool, but the agent did not provide a final answer."
            else:
                final_message_content = str(last_message)

        serializable_output = agent_output.copy()
        if "messages" in serializable_output:
            serializable_output["messages"] = [
                msg.dict() if hasattr(msg, 'dict') else str(msg)
                for msg in serializable_output["messages"]
            ]
        logger.info(f"DEBUG: Full agent_output (final state) for agent '{agent_id}': {json.dumps(serializable_output, indent=2)}")

        logger.info(f"Agent '{agent_id}' generated a final message: {final_message_content}")
        return {"response": final_message_content}
    except Exception as e:
        logger.error(f"Error chatting with agent '{agent_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during agent interaction: {e}")


# --------- Telegram Webhook ----------

@app.post("/telegram/webhook")
async def tg_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received Telegram webhook data: {json.dumps(data, indent=2)}")

        message = data.get("message")
        if not message:
            logger.debug("Webhook data does not contain a 'message' object. Assuming it's a forwarded payload.")
            chat_id = data.get("chat_id")
            user_id = data.get("user_id")
            user_message = data.get("content")
            message_id = data.get("message_id")
            user_name = data.get("user_name")
        else:
            logger.debug("Webhook data contains a 'message' object.")
            chat_id = message.get("chat", {}).get("id")
            user_id = message.get("from", {}).get("id")
            user_message = message.get("text")
            message_id = message.get("message_id")
            user_name = message.get("from", {}).get("username") or message.get("from", {}).get("first_name") or str(user_id)
        
        incoming_bot_id = data.get('bot_id') 

        if not all([chat_id, user_id, user_message, incoming_bot_id]):
            logger.warning(f"Missing essential Telegram message data. Skipping processing. Details: chat_id={chat_id}, user_id={user_id}, user_message={user_message}, bot_id={incoming_bot_id}")
            return {"status": "ok"}

        logger.info(f"Received Telegram message from user {user_id} in chat {chat_id} (via bot {incoming_bot_id}): {user_message}")

        selected_agent_info = None
        logger.debug(f"Attempting to find agent for incoming_bot_id: {incoming_bot_id}")
        all_cached_telegram_ids = {name: info.get('telegram_bot_id') for name, info in agent_manager_instance.get_all_initialized_agents().items()}
        logger.debug(f"Currently cached Telegram bot IDs: {all_cached_telegram_ids}")

        for agent_id, agent_info in agent_manager_instance.get_all_initialized_agents().items():
            cached_telegram_bot_id = agent_info.get("telegram_bot_id")
            logger.debug(f"Checking agent '{agent_info.get('name')}' (ID: {agent_id}) with cached Telegram Bot ID: {cached_telegram_bot_id}. Comparing to incoming bot_id: {incoming_bot_id}")
            
            if agent_info["name"] == "DefaultBot":
                continue 
            
            if agent_info["mcp_client"].tools.get("send_message_telegram") and str(cached_telegram_bot_id) == str(incoming_bot_id): # Ensure string comparison
                selected_agent_info = agent_info
                logger.info(f"Selected agent '{agent_info['name']}' (ID: {agent_id}) for Telegram webhook based on bot ID match.")
                break 

        if not selected_agent_info:
            logger.warning(f"No suitable agent found with Telegram API keys matching bot ID '{incoming_bot_id}' to reply to this message. Message ignored. Available agents' Telegram IDs: {[info.get('telegram_bot_id') for info in agent_manager_instance.get_all_initialized_agents().values() if info.get('telegram_bot_id')]}")
            return {"status": "ignored", "detail": f"No agent configured for Telegram replies via bot ID {incoming_bot_id}."}
        
        agent_executor = selected_agent_info["executor"]
        agent_mcp_client = selected_agent_info["mcp_client"]

        logger.info(f"Invoking agent '{selected_agent_info['name']}' with Telegram message...")
        initial_state = {"messages": [HumanMessage(content=user_message)]}
        agent_output = await agent_executor.ainvoke(initial_state)

        final_message_content = "I'm sorry, I couldn't process that."
        if "messages" in agent_output and agent_output["messages"]:
            last_message = agent_output["messages"][-1]
            if isinstance(last_message, AIMessage):
                final_message_content = last_message.content
            else:
                final_message_content = str(last_message)

        logger.info(f"Agent '{selected_agent_info['name']}' generated Telegram reply: {final_message_content}")

        telegram_tool = agent_mcp_client.tools.get("send_message_telegram")
        if telegram_tool:
            logger.info(f"Using agent '{selected_agent_info['name']}'s own Telegram tool to send reply.")
            send_result = await telegram_tool.ainvoke({ 
                "chat_id": str(chat_id),
                "message": final_message_content
            })
            logger.info(f"Telegram send_message tool call result: {send_result}")
        else:
            logger.error(f"Selected agent '{selected_agent_info['name']}' unexpectedly does not have 'send_message_telegram' tool. Cannot send reply.")

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error processing Telegram webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/discord/receive_message")
async def receive_discord_message(payload: ReceiveDiscordMessageRequest):
    try:
        channel_id = payload.channel_id
        author_id = payload.author_id
        author_name = payload.author_name
        message_content = payload.content
        incoming_bot_id = payload.bot_id 

        logger.info(f"Received Discord message from {author_name} ({author_id}) via bot {incoming_bot_id} in channel {channel_id}: {message_content}")

        selected_agent_info = None
        for agent_id, agent_info in agent_manager_instance.get_all_initialized_agents().items():
            if agent_info["name"] == "DefaultBot":
                continue 
            
            agent_discord_bot_id = agent_info.get("discord_bot_id")
            
            if agent_info["mcp_client"].tools.get("send_message") and str(agent_discord_bot_id) == str(incoming_bot_id): # Ensure string comparison
                selected_agent_info = agent_info
                logger.info(f"Selected agent '{agent_info['name']}' (ID: {agent_id}) for Discord webhook based on bot ID match.")
                break 

        if not selected_agent_info:
            logger.warning(f"No suitable agent found with Discord API keys matching bot ID '{incoming_bot_id}' to reply to this message. Message ignored.")
            return {"status": "ignored", "detail": f"No agent configured for Discord replies via bot ID {incoming_bot_id}."}
        
        agent_executor = selected_agent_info["executor"]
        agent_mcp_client = selected_agent_info["mcp_client"]

        logger.info(f"Invoking agent '{selected_agent_info['name']}' with Discord message...")
        initial_state = {"messages": [HumanMessage(content=message_content)]}
        agent_output = await agent_executor.ainvoke(initial_state)

        final_message_content = "I'm sorry, I couldn't process that."
        if "messages" in agent_output and agent_output["messages"]:
            last_message = agent_output["messages"][-1]
            if isinstance(last_message, AIMessage):
                final_message_content = last_message.content
            else:
                final_message_content = str(last_message)

        logger.info(f"Agent '{selected_agent_info['name']}' generated Discord reply: {final_message_content}")

        discord_tool = agent_mcp_client.tools.get("send_message")
        if discord_tool:
            logger.info(f"Using agent '{selected_agent_info['name']}'s own Discord tool to send reply.")
            send_result = await discord_tool.ainvoke({
                "channel_id": str(channel_id),
                "message": final_message_content
            })
            logger.info(f"Discord send_message tool call result: {send_result}")
        else:
            logger.error(f"Selected agent '{selected_agent_info['name']}' unexpectedly does not have 'send_message' tool. Cannot send reply.")

        return {"status": "ok"}

    except ValidationError as e:
        logger.warning(f"Discord message validation failed: {e.errors()}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Discord message payload: {e.errors()}")
    except Exception as e:
        logger.error(f"Error processing received Discord message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
