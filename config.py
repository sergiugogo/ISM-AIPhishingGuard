"""Configuration management for PhishGuard."""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "PhishGuard API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Model
    model_dir: str = "models/phishguard-model"
    max_length: int = 256
    
    # API Security
    api_key: str = "your-secret-api-key-here"
    allowed_origins: List[str] = ["*"]
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Content Limits
    max_content_length: int = 10000
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Performance
    enable_metrics: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
