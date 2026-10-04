from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read from environment variables, then a .env file in the working directory."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Matches docker-compose defaults; override with DATABASE_URL in .env.
    database_url: str = "postgresql+psycopg://contractqa:contractqa@localhost:5434/contractqa"


def get_settings() -> Settings:
    return Settings()
