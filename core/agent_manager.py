import logging
import os # Added for os.getenv
import asyncio # Added for asyncio.sleep
from typing import Any, Dict, Tuple, Optional, List # Added List for type hinting
from pydantic import Field, PrivateAttr

from langchain_groq import ChatGroq
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from models.agent_config import AgentConfig
from prompts import AGENT_SYSTEM_PROMPT
from langgraph_agents.custom_tool_agent import create_custom_tool_agent

logger = logging.getLogger(__name__)

_initialized_agents: Dict[str, Dict[str, Any]] = {}

def add_initialized_agent(agent_id: str, agent_name: str, executor: Any, mcp_client: MultiServerMCPClient, 
                          discord_bot_id: Optional[str] = None, telegram_bot_id: Optional[str] = None):
    """Adds an initialized agent, its MCP client, and platform-specific bot IDs to the cache."""
    agent_info = {
        "name": agent_name,
        "executor": executor,
        "mcp_client": mcp_client
    }
    if discord_bot_id:
        agent_info["discord_bot_id"] = discord_bot_id
    if telegram_bot_id:
        agent_info["telegram_bot_id"] = telegram_bot_id
        
    _initialized_agents[agent_id] = agent_info
    logger.info(f"Agent '{agent_name}' (ID: {agent_id}) and its MCP client added to cache. Discord Bot ID: {discord_bot_id}, Telegram Bot ID: {telegram_bot_id}")

def get_initialized_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an initialized agent (executor and mcp_client) from the cache."""
    return _initialized_agents.get(agent_id)

def clear_initialized_agents_cache():
    """Clears all initialized agents from the cache."""
    _initialized_agents.clear()
    logger.info("Initialized agents cache cleared.")


class TelegramToolWrapper(BaseTool):
    """
    A wrapper for Telegram tools that injects API credentials into the tool's arguments.
    This allows a single Telegram MCP server to manage multiple Telegram bots.
    """
    _wrapped_tool: BaseTool = PrivateAttr()
    telegram_api_id: int = Field(..., description="Telegram API ID for the bot.")
    telegram_api_hash: str = Field(..., description="Telegram API Hash for the bot.")
    telegram_bot_token: str = Field(..., description="Telegram Bot Token.")

    def __init__(self, wrapped_tool: BaseTool, telegram_api_id: int, telegram_api_hash: str, telegram_bot_token: str, **kwargs: Any):
        super().__init__(
            name=wrapped_tool.name,
            description=wrapped_tool.description,
            args_schema=wrapped_tool.args_schema,
            return_direct=wrapped_tool.return_direct,
            func=wrapped_tool.func,
            coroutine=wrapped_tool.coroutine,
            telegram_api_id=telegram_api_id,
            telegram_api_hash=telegram_api_hash,
            telegram_bot_token=telegram_bot_token,
            **kwargs
        )
        self._wrapped_tool = wrapped_tool

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        all_kwargs = {**kwargs} 
        all_kwargs['telegram_api_id'] = self.telegram_api_id
        all_kwargs['telegram_api_hash'] = self.telegram_api_hash
        all_kwargs['telegram_bot_token'] = self.telegram_bot_token
        
        logger.debug(f"Invoking wrapped Telegram tool '{self.name}' with injected credentials. Final Args: {all_kwargs}")
        return await self._wrapped_tool.ainvoke(all_kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Telegram tools are asynchronous and should use _arun.")


class DiscordToolWrapper(BaseTool):
    """
    A wrapper for Discord tools that injects the bot_id into the tool's arguments.
    This allows a single Discord MCP server to manage multiple Discord bots.
    """
    _wrapped_tool: BaseTool = PrivateAttr()
    bot_id: str = Field(..., description="The Discord bot ID to use for this tool.")

    def __init__(self, wrapped_tool: BaseTool, bot_id: str, **kwargs: Any):
        super().__init__(
            name=wrapped_tool.name,
            description=wrapped_tool.description,
            args_schema=wrapped_tool.args_schema,
            return_direct=wrapped_tool.return_direct,
            func=wrapped_tool.func,
            coroutine=wrapped_tool.coroutine,
            bot_id=bot_id, # Pass to Pydantic
            **kwargs
        )
        self._wrapped_tool = wrapped_tool

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously runs the wrapped tool, injecting the Discord bot_id."""
        all_kwargs = {**kwargs}
        all_kwargs['bot_id'] = self.bot_id # Inject the bot_id
        
        logger.debug(f"Invoking wrapped Discord tool '{self.name}' with injected bot_id: {self.bot_id}. Final Args: {all_kwargs}")
        return await self._wrapped_tool.ainvoke(all_kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Discord tools are asynchronous and should use _arun.")


async def create_dynamic_agent_instance(agent_config: AgentConfig, local_mode: bool) -> Tuple[Any, MultiServerMCPClient, Optional[str], Optional[str]]:
    """
    Dynamically creates and initializes an agent instance based on AgentConfig.
    Returns the compiled agent executor, its associated MCPClient, and fetched bot IDs.
    The MCPClient will include tools based on the provided secrets, ensuring
    the basic toolkit (web, finance, rag) is always present.
    """
    agent_id = agent_config.id
    agent_name = agent_config.name
    logger.info(f"Dynamically initializing agent '{agent_name}' (ID: {agent_id})...")

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=agent_config.secrets.groq_api_key
    )
    logger.info(f"✅ Initialized Groq LLM for agent '{agent_name}' with llama3-8b-8192")
    
    # Helper function to get the correct URL based on local_mode
    def get_mcp_url(service_name: str, port: int, local_mode: bool) -> str:
        if local_mode:
            return f"http://localhost:{port}/mcp/"
        else:
            # For Docker Compose, the service name is the hostname
            return f"http://{service_name}:{port}/mcp/"

    agent_mcp_config = {
        "websearch": {"url": get_mcp_url("web-mcp", 9000, local_mode), "transport": "streamable_http"},
        "finance": {"url": get_mcp_url("finance-mcp", 9001, local_mode), "transport": "streamable_http"},
        "rag": {"url": get_mcp_url("rag-mcp", 9002, local_mode), "transport": "streamable_http"},
    }

    discord_bot_id = None
    telegram_bot_id = None

    # Check for Discord secrets and add Discord MCP if present
    discord_secrets_provided = bool(agent_config.secrets.discord_bot_token)
    if discord_secrets_provided:
        agent_mcp_config["discord"] = {"url": get_mcp_url("discord-mcp", 9004, local_mode), "transport": "streamable_http"}
        logger.info(f"Agent '{agent_name}' will include Discord tools.")
    else:
        logger.info(f"Agent '{agent_name}' does not have Discord bot token. Discord tools will NOT be enabled.")

    # Check for Telegram secrets and add Telegram MCP if present
    telegram_secrets_provided = (
        agent_config.secrets.telegram_bot_token and
        agent_config.secrets.telegram_api_id is not None and # Ensure it's not None
        agent_config.secrets.telegram_api_hash
    )
    if telegram_secrets_provided:
        agent_mcp_config["telegram"] = {"url": get_mcp_url("telegram-mcp", 9003, local_mode), "transport": "streamable_http"}
        logger.info(f"Agent '{agent_name}' will include Telegram tools.")
    else:
        if agent_config.secrets.telegram_bot_token:
            logger.warning(f"Agent '{agent_name}' has Telegram bot token but is missing telegram_api_id or telegram_api_hash. Telegram tools will NOT be enabled.")


    mcp_client = MultiServerMCPClient(agent_mcp_config)
    mcp_client.tools = {} # Initialize mcp_client.tools before populating

    agent_tools_raw = []
    agent_tools_final = []

    # --- Robust Retry Logic for MCP Tools Loading ---
    max_retries = 15
    retry_delay_seconds = 1
    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries}: Loading tools for agent '{agent_name}' from MCP servers at {list(agent_mcp_config.keys())}...")
        try:
            fetched_tools_list = await mcp_client.get_tools()
            if fetched_tools_list:
                agent_tools_raw = list(fetched_tools_list)
                logger.info(f"Successfully fetched tools on attempt {attempt}.")
                break # Exit retry loop on success
            else:
                logger.warning(f"No tools fetched on attempt {attempt}. Retrying...")
        except ExceptionGroup as eg: # Catch ExceptionGroup specifically
            logger.error(f"Error loading tools for agent '{agent_name}' on attempt {attempt}: {eg}", exc_info=True)
            # You can inspect eg.exceptions here if needed for more granular logging
            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay_seconds} seconds before retrying...")
                await asyncio.sleep(retry_delay_seconds)
            else:
                logger.error(f"Failed to load tools after {max_retries} attempts.")
                if not hasattr(mcp_client, 'tools'):
                    mcp_client.tools = {}
                agent_tools_final = []
                # Re-raise the ExceptionGroup if all retries failed
                raise eg 
        except Exception as e: # Catch any other unexpected exceptions
            logger.error(f"Unexpected error loading tools for agent '{agent_name}' on attempt {attempt}: {e}", exc_info=True)
            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay_seconds} seconds before retrying...")
                await asyncio.sleep(retry_delay_seconds)
            else:
                logger.error(f"Failed to load tools after {max_retries} attempts due to unexpected error.")
                if not hasattr(mcp_client, 'tools'):
                    mcp_client.tools = {}
                agent_tools_final = []
                raise # Re-raise the unexpected exception
    
    # If tools were not fetched after all retries, ensure agent_tools_final is empty
    if not agent_tools_raw:
        agent_tools_final = []
    else:
        # Continue with tool processing if tools were fetched
        # First, handle Discord bot registration if token is provided
        if discord_secrets_provided:
            register_discord_tool = next((t for t in agent_tools_raw if t.name == "register_discord_bot"), None)
            if register_discord_tool:
                try:
                    logger.info(f"Calling 'register_discord_bot' for agent '{agent_name}' with token (first 5 chars): {agent_config.secrets.discord_bot_token[:5]}...")
                    # The register_discord_bot tool returns the bot_id
                    discord_bot_id = await register_discord_tool.ainvoke({"bot_token": agent_config.secrets.discord_bot_token})
                    logger.info(f"Successfully registered Discord bot for agent '{agent_name}'. Bot ID: {discord_bot_id}")
                except Exception as e:
                    logger.error(f"Failed to register Discord bot for agent '{agent_name}': {e}", exc_info=True)
                    discord_bot_id = None # Ensure it's None if registration fails
            else:
                logger.warning(f"Agent '{agent_name}' has Discord token but 'register_discord_bot' tool not found. Discord tools will NOT be enabled.")
        
        # Now, process and wrap all tools
        for tool_item in agent_tools_raw:
            if telegram_secrets_provided and tool_item.name in ["send_message_telegram", "get_chat_history", "get_bot_id_telegram"]:
                logger.debug(f"Wrapping Telegram tool '{tool_item.name}' for agent '{agent_name}'.")
                try:
                    telegram_api_id_int = int(agent_config.secrets.telegram_api_id)
                except (ValueError, TypeError):
                    logger.error(f"Invalid or missing telegram_api_id for agent '{agent_name}': {agent_config.secrets.telegram_api_id}. Skipping Telegram tool wrapping.")
                    agent_tools_final.append(tool_item)
                    mcp_client.tools[tool_item.name] = tool_item
                    continue

                wrapped_tool = TelegramToolWrapper(
                    wrapped_tool=tool_item,
                    telegram_api_id=telegram_api_id_int,
                    telegram_api_hash=agent_config.secrets.telegram_api_hash,
                    telegram_bot_token=agent_config.secrets.telegram_bot_token
                )
                agent_tools_final.append(wrapped_tool)
                mcp_client.tools[wrapped_tool.name] = wrapped_tool 
            
            elif discord_bot_id and tool_item.name in ["send_message", "get_channel_messages", "get_bot_id"]:
                logger.debug(f"Wrapping Discord tool '{tool_item.name}' for agent '{agent_name}' with bot ID: {discord_bot_id}.")
                wrapped_tool = DiscordToolWrapper(
                    wrapped_tool=tool_item,
                    bot_id=discord_bot_id
                )
                agent_tools_final.append(wrapped_tool)
                mcp_client.tools[wrapped_tool.name] = wrapped_tool
            
            else:
                agent_tools_final.append(tool_item)
                mcp_client.tools[tool_item.name] = tool_item
        
        if telegram_secrets_provided and "telegram" in agent_mcp_config:
            get_telegram_bot_id_tool = mcp_client.tools.get("get_bot_id_telegram")
            if get_telegram_bot_id_tool:
                try:
                    telegram_bot_id = await get_telegram_bot_id_tool.ainvoke({})
                    logger.info(f"Fetched Telegram Bot ID for agent '{agent_name}': {telegram_bot_id}")
                except Exception as e:
                    logger.warning(f"Failed to fetch Telegram Bot ID for agent '{agent_name}': {e}", exc_info=True)
        

    logger.info(f"🔧 Loaded {len(agent_tools_final)} tools for agent '{agent_name}'. Tools found: {[t.name for t in agent_tools_final]}.")
    logger.info(f"Final number of tools obtained for agent '{agent_name}': {len(agent_tools_final)}")

    system_prompt = AGENT_SYSTEM_PROMPT
    if agent_config.persona:
        system_prompt = f"{system_prompt}\n\nYour persona: {agent_config.persona}"
    if agent_config.bio:
        system_prompt = f"{system_prompt}\n\nYour bio: {agent_config.bio}"
    if agent_config.knowledge:
        system_prompt = f"{system_prompt}\n\nKnowledge: {agent_config.knowledge}" 
    logger.info(f"Using AGENT_SYSTEM_PROMPT (or extended) for agent '{agent_name}'.")

    agent_executor = await create_custom_tool_agent(llm, agent_tools_final, system_prompt, agent_name)

    logger.info(f"🧠 Agent: {agent_name} (ID: {agent_id}) initialized as a custom LangGraph agent with {len(agent_tools_final)} tools.")
    return agent_executor, mcp_client, discord_bot_id, telegram_bot_id