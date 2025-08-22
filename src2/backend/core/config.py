# src2/backend/core/config.py
import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from backend.utils import resource_path_2

class Config:
    """Centralized configuration for RAG + transcription system."""

    def __init__(self, yaml_path: str = resource_path_2("config.yaml"), env_path: str = resource_path_2(".env")):
        # Load .env
        load_dotenv(env_path)

        # Load YAML
        if Path(yaml_path).exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                self.yaml_config = yaml.safe_load(f) or {}
            
        else:
            print(f"YAML PATH DOESN'T EXIST")
            self.yaml_config = {}

        # === ENV (secrets only) ===
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        self.GROK_API_KEY = os.getenv("GROQ_API_KEY", "")

        # === YAML ===
        self.milvus = self.yaml_config.get("milvus", {})
        self.llm = self.yaml_config.get("llm", {})
        self.chunking = self.yaml_config.get("chunking", {})
        self.collection = self.yaml_config.get("collection", {})
        self.retrieval = self.yaml_config.get("retrieval", {})
        self.transcription = self.yaml_config.get("transcription", {})
        self.database = self.yaml_config.get("database",{})

def load_config():
    global _config
    if "_config" not in globals():
        globals()["_config"] = Config()
    return globals()["_config"]
