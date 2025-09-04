from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Loads and validates application settings from environment variables.
    """
    openai_api_key: str

    class Config:
        env_file = ".env"

# Create a single instance to be used across the application
settings = Settings()