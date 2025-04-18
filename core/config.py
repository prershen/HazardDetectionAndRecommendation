from pydantic.v1 import BaseSettings
from os import environ

class Settings(BaseSettings):
    mongo_uri: str = environ.get("MONGO_URI")
    database_name: str = environ.get("DATABASE_NAME")
    jwt_secret: str = environ.get("JWT_SECRET")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 2880 # 2 days
    gemini_api_key: str = environ.get("GEMINI_API_KEY")

    class Config:
        env_file = ".env"

settings = Settings()