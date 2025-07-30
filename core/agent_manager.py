import logging
import asyncio
import os
import json 
import uuid 
from typing import Any, Dict, Tuple, Optional, List
from pydantic import Field, PrivateAttr

from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from models.agent_config import AgentConfig, AgentSecrets, Settings 
from prompts import AGENT_SYSTEM_PROMPT
from langgraph_agents.custom_tool_agent import create_custom_tool_agent
from llm_factory import create_llm 
from db.sqlite_manager import SQLiteManager


logger = logging.getLogger(__name__)

DEFAULT_CHARACTER_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "default.character.json")


def _load_default_agent_config_from_file() -> Optional[AgentConfig]:
    """
    Loads the default agent configuration from the default.character.json file
    and maps it to the AgentConfig Pydantic model.
    """
    if not os.path.exists(DEFAULT_CHARACTER_CONFIG_PATH):
        logger.warning(f"Default character config file not found at {DEFAULT_CHARACTER_CONFIG_PATH}. Skipping default agent creation from file.")
        return None

    try:
        config_data = None
        try:
            with open(DEFAULT_CHARACTER_CONFIG_PATH, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {DEFAULT_CHARACTER_CONFIG_PATH}. Trying latin-1 encoding.")
            with open(DEFAULT_CHARACTER_CONFIG_PATH, 'r', encoding='latin-1') as f:
                config_data = json.load(f)

        if config_data is None:
            logger.error(f"Could not load data from {DEFAULT_CHARACTER_CONFIG_PATH} with utf-8 or latin-1 encoding.")
            return None

        settings_data = config_data.get("settings", {})
        secrets_from_json = settings_data.get("secrets", {})
        logger.debug(f"[_load_default_agent_config_from_file] Raw secrets from JSON: {secrets_from_json}")
        
        voice_settings = settings_data.get("voice", {})

        # --- SIMPLIFIED SECRET PARSING ---
        agent_secrets_instance = AgentSecrets(**secrets_from_json)
        logger.debug(f"[_load_default_agent_config_from_file] Parsed AgentSecrets: {agent_secrets_instance.model_dump_json(exclude_none=True)}")

        # Create Settings instance
        settings_instance = Settings(
            model=settings_data.get("model", "llama3-8b-8192"),
            temperature=settings_data.get("temperature", 0.7),
            maxTokens=settings_data.get("maxTokens", 15000),
            secrets=agent_secrets_instance, 
            voice=voice_settings if voice_settings else None
        )

        # Create AgentConfig instance
        default_agent_config = AgentConfig(
            id=str(uuid.uuid4()), 
            name=config_data.get("name", "DefaultBot"),
            modelProvider=config_data.get("modelProvider", "groq"), 
            settings=settings_instance, 
            system=config_data.get("system", AGENT_SYSTEM_PROMPT), 
            bio=config_data.get("bio", []),
            lore=config_data.get("lore", []),
            knowledge=config_data.get("knowledge", []), 
            messageExamples=config_data.get("messageExamples"), 
            style=config_data.get("style")
        )
        logger.info(f"Successfully loaded default agent config from {DEFAULT_CHARACTER_CONFIG_PATH}")
        return default_agent_config

    except FileNotFoundError:
        logger.error(f"Failed to load default character config: {DEFAULT_CHARACTER_CONFIG_PATH} not found.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse default character config JSON from {DEFAULT_CHARACTER_CONFIG_PATH}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading default character config: {e}", exc_info=True)
        return None


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
            bot_id=bot_id, 
            **kwargs
        )
        self._wrapped_tool = wrapped_tool

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously runs the wrapped tool, injecting the Discord bot_id."""
        all_kwargs = {**kwargs}
        all_kwargs['bot_id'] = self.bot_id 
        
        logger.debug(f"Invoking wrapped Discord tool '{self.name}' with injected bot_id: {self.bot_id}. Final Args: {all_kwargs}")
        return await self._wrapped_tool.ainvoke(all_kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Discord tools are asynchronous and should use _arun.")


class AgentManager:
    """
    Manages the lifecycle, initialization, and caching of AI agents.
    This class encapsulates the _initialized_agents dictionary and provides methods
    to interact with it, as well as handle dynamic agent creation and shutdown.
    """
    def __init__(self, db_path: str):
        self._initialized_agents: Dict[str, Dict[str, Any]] = {}
        self.db_manager = SQLiteManager(db_path) 

    def add_initialized_agent(self, agent_id: str, agent_name: str, executor: Any, mcp_client: MultiServerMCPClient, 
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
            
        self._initialized_agents[agent_id] = agent_info
        logger.info(f"Agent '{agent_name}' (ID: {agent_id}) and its MCP client added to cache. Discord Bot ID: {discord_bot_id}, Telegram Bot ID: {telegram_bot_id}")

    def get_initialized_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an initialized agent (executor and mcp_client) from the cache."""
        return self._initialized_agents.get(agent_id)

    def get_all_initialized_agents(self) -> Dict[str, Dict[str, Any]]:
        """Returns all initialized agents from the cache."""
        return self._initialized_agents

    async def shutdown_all_agents(self):
        """Shuts down all initialized agents and their components."""
        logger.info("Shutting down all agents...")
        for agent_id, agent_info in list(self._initialized_agents.items()):
            await self.shutdown_specific_agent(agent_id)
        logger.info("All agents shut down and cache cleared.")

    async def shutdown_specific_agent(self, agent_id: str):
        """Shuts down a specific agent and removes it from the cache."""
        agent_info = self._initialized_agents.pop(agent_id, None)
        if agent_info:
            mcp_client = agent_info.get("mcp_client")
            if mcp_client:
                await mcp_client.close()
                logger.info(f"MCP Client for agent {agent_id} closed.")
            
            logger.info(f"Agent {agent_id} removed from cache.")
        else:
            logger.warning(f"Attempted to shut down agent {agent_id}, but it was not found in cache.")

    async def initialize_all_agents_from_db(self, local_mode: bool):
        """
        Initializes or re-initializes all agents found in the database.
        This method is called during application startup.
        """
        existing_configs = await self.db_manager.get_all_agent_configs()

        if not existing_configs:
            logger.info("No existing agent configurations found. Attempting to load default agent from file.")
            default_agent_config = _load_default_agent_config_from_file() 

            if default_agent_config:
                await self.db_manager.save_agent_config(default_agent_config)
                logger.info(f"Default agent '{default_agent_config.name}' saved to DB with ID: {default_agent_config.id}.")
                existing_configs = await self.db_manager.get_all_agent_configs() 
            else:
                logger.error("Could not load or create a default agent configuration. No agents will be initialized.")
                return 
            
        for config in existing_configs:
            try:
                if not config.id:
                    config.id = str(uuid.uuid4()) 
                    logger.warning(f"Agent config for '{config.name}' has no ID. Generated new ID: {config.id}")
                    await self.db_manager.update_agent_config(config) 

                executor, mcp_client, discord_bot_id, telegram_bot_id = \
                    await self.create_dynamic_agent_instance(config, local_mode)
                
                self.add_initialized_agent( 
                    config.id,
                    config.name,
                    executor,
                    mcp_client,
                    discord_bot_id=discord_bot_id,
                    telegram_bot_id=telegram_bot_id
                )
            except Exception as e:
                logger.error(f"Failed to re-initialize agent '{config.name}' (ID: {config.id}): {e}", exc_info=True)

        logger.info(f"Finished initializing {len(self._initialized_agents)} agent(s).")
            
    async def create_dynamic_agent_instance(self, agent_config: AgentConfig, local_mode: bool) -> Tuple[Any, MultiServerMCPClient, Optional[str], Optional[str]]:
        """
        Dynamically creates and initializes an agent instance based on AgentConfig.
        Returns the compiled agent executor (LangGraph runnable), its associated MCPClient,
        and fetched bot IDs.
        """
        agent_id = agent_config.id
        agent_name = agent_config.name
        llm_model_provider = agent_config.modelProvider
        llm_model_name = agent_config.settings.model
        llm_temperature = agent_config.settings.temperature
        llm_max_tokens = agent_config.settings.maxTokens
        llm_secrets = agent_config.settings.secrets 

        agent_bio = agent_config.bio
        agent_persona = agent_config.system 
        agent_knowledge = agent_config.knowledge 
        agent_lore = agent_config.lore 
        agent_style = agent_config.style 
        agent_message_examples = agent_config.messageExamples 

        logger.info(f"Dynamically initializing agent '{agent_name}' (ID: {agent_id})...")

        # --- LLM Initialization using the factory ---
        llm_api_key = None
        if llm_model_provider == "groq":
            llm_api_key = llm_secrets.groq_api_key or os.getenv("GROQ_API_KEY")
        elif llm_model_provider == "google":
            llm_api_key = llm_secrets.google_api_key or os.getenv("GOOGLE_API_KEY")
        elif llm_model_provider == "openai":
            llm_api_key = llm_secrets.openai_api_key or os.getenv("OPENAI_API_KEY")
        elif llm_model_provider == "anthropic":
            llm_api_key = llm_secrets.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        try:
            llm = create_llm(
                provider=llm_model_provider,
                api_key=llm_api_key,
                model=llm_model_name,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens
            )
            logger.info(f"✅ Initialized LLM for agent '{agent_name}': Provider={llm_model_provider}, Model={llm_model_name}")
        except ValueError as e:
            logger.error(f"Failed to initialize LLM for agent '{agent_name}': {e}", exc_info=True)
            raise 

        # --- MCP Server Configuration (using Docker Compose service names for non-local) ---
        agent_mcp_config = {
            "multi_search": {"url": "http://localhost:9000/mcp/", "transport": "streamable_http"}, 
            "finance": {"url": "http://localhost:9001/mcp/", "transport": "streamable_http"}, 
            "rag": {"url": "http://localhost:9002/mcp/", "transport": "streamable_http"}, 
        }
        
        if not local_mode:
            agent_mcp_config["multi_search"]["url"] = "http://web-mcp:9000/mcp/"
            agent_mcp_config["finance"]["url"] = "http://finance-mcp:9000/mcp/"
            agent_mcp_config["rag"]["url"] = "http://rag-mcp:9000/mcp/"


        discord_bot_id = None
        telegram_bot_id = None

        discord_token = llm_secrets.discord_bot_token 
        discord_secrets_provided = bool(discord_token)
        if discord_secrets_provided:
            if local_mode:
                agent_mcp_config["discord"] = {"url": "http://localhost:9004/mcp/", "transport": "streamable_http"}
            else:
                agent_mcp_config["discord"] = {"url": "http://discord-mcp:9000/mcp/", "transport": "streamable_http"}
            logger.info(f"Agent '{agent_name}' will include Discord tools.")
        else:
            logger.info(f"Agent '{agent_name}' does not have Discord bot token. Discord tools will NOT be enabled.")

        telegram_token = llm_secrets.telegram_bot_token 
        telegram_api_id = llm_secrets.telegram_api_id 
        telegram_api_hash = llm_secrets.telegram_api_hash

        telegram_secrets_provided = (
            telegram_token and
            telegram_api_id is not None and 
            telegram_api_hash
        )
        if telegram_secrets_provided:
            if local_mode:
                agent_mcp_config["telegram"] = {"url": "http://localhost:9003/mcp/", "transport": "streamable_http"}
            else:
                agent_mcp_config["telegram"] = {"url": "http://telegram-mcp:9000/mcp/", "transport": "streamable_http"}
            logger.info(f"Agent '{agent_name}' will include Telegram tools.")
        else:
            if telegram_token:
                logger.warning(f"Agent '{agent_name}' has Telegram bot token but is missing telegram_api_id or telegram_api_hash. Telegram tools will NOT be enabled.")


        mcp_client = MultiServerMCPClient(agent_mcp_config)
        mcp_client.tools = {} 

        agent_tools_raw = []
        agent_tools_final = []

        logger.info(f"Attempting to load tools for agent '{agent_name}' from MCP servers: {list(agent_mcp_config.keys())}...")
        
        # --- RETRY MECHANISM ---
        max_attempts = 12  
        base_delay = 2    
        for attempt in range(1, max_attempts + 1): 
            try:
                fetched_tools_list = await mcp_client.get_tools()
                if fetched_tools_list:
                    agent_tools_raw = list(fetched_tools_list)
                    logger.info(f"Successfully fetched {len(agent_tools_raw)} raw tools on attempt {attempt}.")
                    break 
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_attempts} failed to fetch tools for agent '{agent_name}': {e}")
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1)) 
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else: 
                    logger.error(f"Failed to fetch tools for agent '{agent_name}' after {max_attempts} attempts. Configured MCP servers might be down or inaccessible.")
                    # Ensure mcp_client.tools is initialized even if tool fetching fails completely
                    if not hasattr(mcp_client, 'tools'):
                        mcp_client.tools = {}
                    agent_tools_final = [] 
                    raise 

        if agent_tools_raw:
            if discord_secrets_provided:
                register_discord_tool = next((t for t in agent_tools_raw if t.name == "register_discord_bot"), None)
                if register_discord_tool:
                    try:
                        logger.info(f"Calling 'register_discord_bot' for agent '{agent_name}' with token (first 5 chars): {discord_token[:5]}...")
                        discord_bot_id = await register_discord_tool.ainvoke({"bot_token": discord_token})
                        logger.info(f"Successfully registered Discord bot for agent '{agent_name}'. Bot ID: {discord_bot_id}")
                    except Exception as e:
                        logger.error(f"Failed to register Discord bot for agent '{agent_name}': {e}", exc_info=True)
                        discord_bot_id = None 
                else:
                    logger.warning(f"Agent '{agent_name}' has Discord token but 'register_discord_bot' tool not found. Discord tools will NOT be enabled.")
                
            for tool_item in agent_tools_raw: 
                if telegram_secrets_provided and tool_item.name in ["send_message_telegram", "get_chat_history", "get_bot_id_telegram"]:
                    logger.debug(f"Wrapping Telegram tool '{tool_item.name}' for agent '{agent_name}'.")
                    try:

                        telegram_api_id_int = int(telegram_api_id) if telegram_api_id is not None else 0 
                    except (ValueError, TypeError): 
                        logger.error(f"Invalid or missing telegram_api_id for agent '{agent_name}': {telegram_api_id}. Skipping Telegram tool wrapping.")
                        # If conversion fails, don't wrap, just add the raw tool if it's not a bot tool
                        if tool_item.name not in ["send_message_telegram", "get_chat_history", "get_bot_id_telegram"]:
                             agent_tools_final.append(tool_item)
                             mcp_client.tools[tool_item.name] = tool_item
                        continue 

                    wrapped_tool = TelegramToolWrapper(
                        wrapped_tool=tool_item,
                        telegram_api_id=telegram_api_id_int,
                        telegram_api_hash=telegram_api_hash,
                        telegram_bot_token=telegram_token
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
        else:
            logger.warning(f"No raw tools fetched for agent '{agent_name}'. Agent will operate without tools.")
            # Ensure mcp_client.tools is an empty dict if no tools were fetched
            if not hasattr(mcp_client, 'tools'):
                mcp_client.tools = {}


        logger.info(f"🔧 Loaded {len(agent_tools_final)} tools for agent '{agent_name}'. Tools found: {[t.name for t in agent_tools_final]}.")
        logger.info(f"Final number of tools obtained for agent '{agent_name}': {len(agent_tools_final)}")

        system_prompt = AGENT_SYSTEM_PROMPT 
        if agent_persona: 
            system_prompt = f"{system_prompt}\n\nYour persona: {agent_persona}"
        if agent_bio: 
            system_prompt = f"{system_prompt}\n\nYour bio: {'\n'.join(agent_bio)}"
        if agent_knowledge: 
            system_prompt = f"{system_prompt}\n\nKnowledge: {'\n'.join(agent_knowledge)}" 
        if agent_lore: 
            system_prompt = f"{system_prompt}\n\nLore: {'\n'.join(agent_lore)}" 
        if agent_style:
            style_str = json.dumps(agent_style, indent=2) if isinstance(agent_style, dict) else str(agent_style)
            system_prompt = f"{system_prompt}\n\nStyle: {style_str}" 
        if agent_message_examples:
            examples_str = json.dumps(agent_message_examples, indent=2)
            system_prompt = f"{system_prompt}\n\nMessage Examples:\n{examples_str}" 

        logger.info(f"Using AGENT_SYSTEM_PROMPT for agent '{agent_name}'.") 

        agent_executor = await create_custom_tool_agent(llm, agent_tools_final, system_prompt, agent_name)

        logger.info(f"🧠 Agent: {agent_name} (ID: {agent_id}) initialized as a custom LangGraph agent with {len(agent_tools_final)} tools.")
        
        return agent_executor, mcp_client, discord_bot_id, telegram_bot_id
