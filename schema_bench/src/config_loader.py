import yaml
import os
from pydantic import BaseModel, Field
from typing import Optional

class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.1
    max_retries: int = 3
    api_key_env_var: str = "OPENAI_API_KEY"

class EvaluationConfig(BaseModel):
    output_dir: str = "results"
    max_scenarios: Optional[int] = None
    save_responses: bool = True

class ReportingConfig(BaseModel):
    format: str = "csv"

class AppConfig(BaseModel):
    llm: LLMConfig
    evaluation: EvaluationConfig
    reporting: ReportingConfig

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return AppConfig(**config_dict)
