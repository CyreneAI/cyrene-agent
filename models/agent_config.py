from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

class AgentSecrets(BaseModel):
    """Pydantic model for storing API keys and secrets for an agent's tools."""
    discord_bot_token: Optional[str] = None
    telegram_api_id: Optional[int] = None
    telegram_api_hash: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    serpapi_api_key: Optional[str] = None
    newsapi_org_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    quandl_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None 
    google_api_key: Optional[str] = None 
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

class Settings(BaseModel):
    """Pydantic model for the nested 'settings' object in the agent config."""
    model: Optional[str] = Field(None, description="The specific LLM model name to use (e.g., 'llama3-70b-8192', 'gemini-pro', 'gpt-4').")
    temperature: float = Field(0.7, description="Temperature for LLM generation.")
    maxTokens: int = Field(8192, description="Maximum number of tokens for LLM generation.")
    secrets: AgentSecrets = Field(default_factory=AgentSecrets)
    voice: Optional[Dict[str, str]] = Field(None, description="Voice model settings.")

    class Config:
        populate_by_name = True
        extra = "forbid"

class AgentConfig(BaseModel):
    """Pydantic model for an agent's overall configuration, matching the JSON structure."""
    id: Optional[str] = None 
    name: str
    modelProvider: str = Field(..., description="The provider of the LLM (e.g., 'anthropic', 'groq', 'google', 'openai', 'ollama').")
    
    settings: Settings = Field(..., description="LLM and other settings for the agent.") # Nested Settings model

    system: Optional[str] = Field(None, description="The persona for the agent.")

    bio: Optional[List[str]] = Field(None, description="A list of biography points.")
    lore: Optional[List[str]] = Field(None, description="Lore and background information as a list.")
    knowledge: Optional[List[str]] = Field(None, description="Specific knowledge points as a list.")

  
    messageExamples: Optional[List[List[Dict[str, Any]]]] = Field(None, description="Examples of messages for the agent.")
    style: Optional[Dict[str, List[str]]] = Field(None, description="Stylistic guidelines for the agent's responses.")

    class Config:
        populate_by_name = True 
        extra = "forbid" 