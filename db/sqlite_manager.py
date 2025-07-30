import sqlite3
import json
import asyncio
import logging
from typing import Optional, List, Tuple, Any

from models.agent_config import AgentConfig, AgentSecrets, Settings

logger = logging.getLogger(__name__)

class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    def _get_db_connection(self):
        """Establishes a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self):
        """Ensures the necessary table exists in the database and handles migrations."""
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()

            create_agents_table_sql = """
            CREATE TABLE IF NOT EXISTS agent_configs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                llm_model_name TEXT,       -- Made nullable
                llm_temperature REAL NOT NULL,
                llm_max_tokens INTEGER NOT NULL,
                persona TEXT,              -- Made nullable
                bio TEXT,
                lore TEXT,
                knowledge TEXT,
                message_examples TEXT,     -- Stored as JSON string
                style TEXT,                -- Stored as JSON string
                secrets TEXT,              -- Stored as JSON string
                voice_settings TEXT        -- Stored as JSON string
            );
            """
            cursor.execute(create_agents_table_sql)
            conn.commit()

            # --- Schema Migration Logic (for existing databases) ---
            cursor.execute("PRAGMA table_info(agent_configs);")
            columns = [col[1] for col in cursor.fetchall()]

            if 'persona' not in columns:
                cursor.execute("ALTER TABLE agent_configs ADD COLUMN persona TEXT;")
                conn.commit()
                logger.info("Added 'persona' column to agent_configs table.")
            
            if 'llm_model_name' not in columns:
                cursor.execute("ALTER TABLE agent_configs ADD COLUMN llm_model_name TEXT;")
                conn.commit()
                logger.info("Added 'llm_model_name' column to agent_configs table.")


        except sqlite3.Error as e:
            logger.error(f"SQLite error during table creation/migration: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    async def save_agent_config(self, agent_config: AgentConfig) -> str:
        """Saves a new agent configuration to the database (upsert)."""
        def _save():
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                
                secrets_json = agent_config.settings.secrets.model_dump_json(exclude_none=True) if agent_config.settings.secrets else "{}"
                message_examples_json = json.dumps(agent_config.messageExamples) if agent_config.messageExamples is not None else "[]"
                style_json = json.dumps(agent_config.style) if agent_config.style is not None else "{}"
                voice_settings_json = json.dumps(agent_config.settings.voice) if agent_config.settings.voice is not None else None # Can be None

                bio_json = json.dumps(agent_config.bio) if agent_config.bio is not None else "[]"
                lore_json = json.dumps(agent_config.lore) if agent_config.lore is not None else "[]"
                knowledge_json = json.dumps(agent_config.knowledge) if agent_config.knowledge is not None else "[]"

                cursor.execute("""
                    INSERT OR REPLACE INTO agent_configs ( -- Use INSERT OR REPLACE for upsert
                        id, name, model_provider, llm_model_name, llm_temperature, 
                        llm_max_tokens, persona, bio, lore, knowledge, 
                        message_examples, style, secrets, voice_settings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_config.id,
                    agent_config.name,
                    agent_config.modelProvider, 
                    agent_config.settings.model, 
                    agent_config.settings.temperature,
                    agent_config.settings.maxTokens, 
                    agent_config.system, 
                    bio_json, 
                    lore_json, 
                    knowledge_json, 
                    message_examples_json,
                    style_json,
                    secrets_json,
                    voice_settings_json
                ))
                conn.commit()
                logger.info(f"Saved agent config '{agent_config.name}' with ID '{agent_config.id}' to SQLite.")
                return agent_config.id
            except sqlite3.Error as e: 
                logger.error(f"SQLite error saving agent config: {e}", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Error saving agent config: {e}", exc_info=True)
                raise
            finally:
                conn.close()
        
        return await asyncio.to_thread(_save)

    async def update_agent_config(self, agent_config: AgentConfig):
        """Updates an existing agent configuration in the database."""
        def _update():
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                
                secrets_json = agent_config.settings.secrets.model_dump_json(exclude_none=True) if agent_config.settings.secrets else "{}"
                message_examples_json = json.dumps(agent_config.messageExamples) if agent_config.messageExamples is not None else "[]"
                style_json = json.dumps(agent_config.style) if agent_config.style is not None else "{}"
                voice_settings_json = json.dumps(agent_config.settings.voice) if agent_config.settings.voice is not None else None 

                bio_json = json.dumps(agent_config.bio) if agent_config.bio is not None else "[]"
                lore_json = json.dumps(agent_config.lore) if agent_config.lore is not None else "[]"
                knowledge_json = json.dumps(agent_config.knowledge) if agent_config.knowledge is not None else "[]"

                cursor.execute("""
                    UPDATE agent_configs SET
                        name = ?,
                        model_provider = ?,
                        llm_model_name = ?,
                        llm_temperature = ?,
                        llm_max_tokens = ?,
                        persona = ?,
                        bio = ?,
                        lore = ?,
                        knowledge = ?,
                        message_examples = ?,
                        style = ?,
                        secrets = ?,
                        voice_settings = ?
                    WHERE id = ?
                """, (
                    agent_config.name,
                    agent_config.modelProvider,
                    agent_config.settings.model,
                    agent_config.settings.temperature,
                    agent_config.settings.maxTokens,
                    agent_config.system,
                    bio_json,
                    lore_json,
                    knowledge_json,
                    message_examples_json,
                    style_json,
                    secrets_json,
                    voice_settings_json,
                    agent_config.id
                ))
                conn.commit()
                logger.info(f"Updated agent config '{agent_config.name}' with ID '{agent_config.id}' in SQLite.")
                return agent_config.id
            except Exception as e:
                logger.error(f"Error updating agent config: {e}", exc_info=True)
                raise
            finally:
                conn.close()

        return await asyncio.to_thread(_update)


    async def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Retrieves an agent configuration by ID."""
        def _get():
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM agent_configs WHERE id = ?", (agent_id,))
                row = cursor.fetchone()
                if row:
                    return self._load_agent_config_from_row(row, cursor.description) 
                return None
            except Exception as e:
                logger.error(f"Error retrieving agent config for ID {agent_id}: {e}", exc_info=True)
                return None
            finally:
                conn.close()
        return await asyncio.to_thread(_get)

    async def get_agent_config_by_name(self, agent_name: str) -> Optional[AgentConfig]:
        """
        Retrieves a single agent's configuration from the database by name.
        """
        def _get_by_name():
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM agent_configs WHERE name = ?", (agent_name,))
                row = cursor.fetchone()
                if row:
                    return self._load_agent_config_from_row(row, cursor.description) 
                return None
            except Exception as e:
                logger.error(f"Error retrieving agent config by name '{agent_name}': {e}", exc_info=True)
                return None
            finally:
                conn.close()
        return await asyncio.to_thread(_get_by_name)

    async def get_all_agent_configs(self) -> List[AgentConfig]:
        """Retrieves all agent configurations from the database."""
        def _get_all():
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM agent_configs")
                rows = cursor.fetchall()
                configs = []
                for row in rows:
                    try:
                        configs.append(self._load_agent_config_from_row(row, cursor.description))
                    except Exception as e:
                        logger.error(f"Error parsing agent config from DB for ID {row[0]}: {e}", exc_info=True)
                logger.info(f"Retrieved {len(configs)} agent configs from SQLite.")
                return configs
            except Exception as e:
                logger.error(f"Error retrieving all agent configs: {e}", exc_info=True)
                return []
            finally:
                conn.close()
        return await asyncio.to_thread(_get_all)

    def _load_agent_config_from_row(self, row: Tuple, cursor_description: List[Tuple]) -> AgentConfig:
        """Helper to load AgentConfig from a database row using column names."""
        # Create a dictionary mapping column names to their values
        column_names = [description[0] for description in cursor_description]
        row_dict = dict(zip(column_names, row))

        # Helper function for robust JSON loading
        def safe_json_load(json_str: Optional[str], default_val: Any):
            if json_str is None or (isinstance(json_str, str) and not json_str.strip()): 
                return default_val
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON string '{json_str}': {e}", exc_info=True)
                return default_val

        # Extract values using column names
        id = row_dict.get('id')
        name = row_dict.get('name')
        model_provider = row_dict.get('model_provider')
        llm_model_name = row_dict.get('llm_model_name') 
        llm_temperature = row_dict.get('llm_temperature')
        llm_max_tokens = row_dict.get('llm_max_tokens')
        persona_db = row_dict.get('persona')
        bio_json = row_dict.get('bio')
        lore_json = row_dict.get('lore')
        knowledge_json = row_dict.get('knowledge')
        message_examples_json = row_dict.get('message_examples')
        style_json = row_dict.get('style')
        secrets_json = row_dict.get('secrets')
        voice_settings_json = row_dict.get('voice_settings')

        # Parse JSON strings back to Python objects using the helper
        secrets_data = safe_json_load(secrets_json, {})
        message_examples_data = safe_json_load(message_examples_json, [])
        style_data = safe_json_load(style_json, {})
        voice_settings_data = safe_json_load(voice_settings_json, None) 
        bio_data = safe_json_load(bio_json, [])
        lore_data = safe_json_load(lore_json, [])
        knowledge_data = safe_json_load(knowledge_json, [])

        agent_secrets_instance = AgentSecrets(**secrets_data)

        settings_instance = Settings(
            model=llm_model_name, 
            temperature=llm_temperature,
            maxTokens=llm_max_tokens,
            secrets=agent_secrets_instance,
            voice=voice_settings_data
        )

        return AgentConfig(
            id=id,
            name=name,
            modelProvider=model_provider,
            settings=settings_instance,
            system=persona_db, 
            bio=bio_data, 
            lore=lore_data, 
            knowledge=knowledge_data, 
            messageExamples=message_examples_data,
            style=style_data
        )

    async def delete_agent_config(self, agent_id: str):
        """Deletes an agent configuration by ID."""
        def _delete():
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM agent_configs WHERE id = ?", (agent_id,))
                conn.commit()
                logger.info(f"Deleted agent config with ID '{agent_id}' from SQLite.")
            except Exception as e:
                logger.error(f"Error deleting agent config for ID {agent_id}: {e}", exc_info=True)
                raise
            finally:
                conn.close()
        return await asyncio.to_thread(_delete)

    def close(self):
        """Placeholder for closing connections if necessary. SQLite connections are typically closed per operation."""
        pass
