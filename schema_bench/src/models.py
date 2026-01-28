from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BenchmarkScenario(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str  # "Easy", "Hard", "Recursive", "Polymorphic"
    track: str # "Structural", "Constraint", "Ambiguity"
    prompt_template: str
    required_constraints: List[str] = Field(default_factory=list) # e.g., ["minimum_age", "email_format"]
    gold_keys: List[str] = Field(default_factory=list) # Expected keys like "user_id", "email"

class EvaluationResult(BaseModel):
    scenario_id: str
    syntax_valid: bool = False
    meta_schema_valid: bool = False
    self_consistent: bool = False
    constraint_recall_score: float = 0.0
    specificity_score: float = 0.0
    errors: List[str] = Field(default_factory=list)
