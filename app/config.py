"""Configuration settings for the application."""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_embedding_model: str = "text-embedding-3-large"
    
    # Qdrant Configuration
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "premed_questions")
    
    # Similarity Threshold
    similarity_threshold: float = 0.75
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

