import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from typing import Optional
from langchain_anthropic import ChatAnthropic

import logging

logger = logging.getLogger(__name__)

def create_llm(provider: str, api_key: Optional[str] = None, model: Optional[str] = None,temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,):
    """
    Factory function to create an LLM instance based on the provider.

    :param provider: The name of the LLM provider (e.g., "groq", "google", "openai", "ollama").
    :param api_key: The API key for the chosen provider. Required for most cloud providers.
    :param model: The specific model name to use (e.g., "llama3-70b-8192", "gemini-pro", "gpt-4").
                  If None, a default model for the provider will be used.
    :return: An initialized LangChain LLM instance.
    :raises ValueError: If the provider is unsupported or a required API key is missing.
    """
    provider = provider.lower()
    
    if provider == "groq":
        if not api_key:
            logger.error("Groq API key is required but not provided.")
            raise ValueError("`api_key` is required for Groq.")
        return ChatGroq(
            model=model or "llama3-8b-8192", # Default to 8b for general use
            temperature=temperature,
            max_tokens=max_tokens,            
            api_key=api_key,
           
        )
    elif provider == "google":
        if not api_key:
            logger.error("Google API key is required but not provided.")
            raise ValueError("`api_key` is required for Google Generative AI.")
        return ChatGoogleGenerativeAI(
            model=model or "gemini-pro", # Updated default model
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
           
        )
    elif provider == "openai":
        if not api_key:
            logger.error("OpenAI API key is required but not provided.")
            raise ValueError("`OPENAI api_key` is required for OpenAI.")
        return ChatOpenAI(
            model_name=model or "gpt-3.5-turbo",
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            
        )
    elif provider == "anthropic": 
        if not api_key:
            logger.error("Anthropic API key is required but not provided.")
            raise ValueError("ANTHROPIC_API_KEY must be provided for Anthropic LLM.")
        return ChatAnthropic(
            model_name=model or "claude-3-opus-20240229", 
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=api_key
        )    

    elif provider == "ollama":
        logger.info(f"Creating Ollama LLM with model: {model or 'llama3'}")
        return ChatOllama(model=model or "llama3", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    

    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")