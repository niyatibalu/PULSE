# app/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_path: str = Field(default="data/accidents.csv")
    model_path: str = Field(default="models/ensemble.joblib")

    # Grid settings
    lat_bins: int = Field(default=20)
    lon_bins: int = Field(default=20)

    # Ensemble / uncertainty
    n_models: int = Field(default=25)
    random_seed: int = Field(default=42)

    # Feature store warmup (hours)
    warmup_hours: int = Field(default=48)

    # API
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)


settings = Settings()