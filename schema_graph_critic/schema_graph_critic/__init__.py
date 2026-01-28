"""
SchemaGraph Critic: A Neuro-Symbolic middleware for validating LLM-generated JSON Schemas.

Uses a Heterogeneous Graph Transformer (HGT) to detect structural logic errors
that standard validators miss.
"""

__version__ = "0.1.0"

from .parser import SchemaGraphParser
from .model import SchemaGNN, HomogeneousSchemaGNN, NoSemanticSchemaGNN, SchemaGNNLoss, create_model
from .corruptor import SchemaCorruptor
from .translator import FeedbackTranslator
from .trainer import Trainer, TrainingConfig
from .evaluate import Evaluator, EvaluationResult, evaluate_checkpoint
from .baselines import (
    CombinedBaseline,
    ReferenceChecker,
    CycleDetector,
    ConstraintChecker,
    StandardValidator,
    run_baseline_evaluation,
)
from .llm_pipeline import (
    SchemaRefinementPipeline,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    MockProvider,
    run_comparison_experiment,
)

__all__ = [
    # Core
    "SchemaGraphParser",
    "SchemaGNN",
    "SchemaCorruptor",
    "FeedbackTranslator",
    # Model variants (for ablation)
    "HomogeneousSchemaGNN",
    "NoSemanticSchemaGNN",
    "SchemaGNNLoss",
    "create_model",
    # Training
    "Trainer",
    "TrainingConfig",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "evaluate_checkpoint",
    # Baselines
    "CombinedBaseline",
    "ReferenceChecker",
    "CycleDetector",
    "ConstraintChecker",
    "StandardValidator",
    "run_baseline_evaluation",
    # LLM Pipeline
    "SchemaRefinementPipeline",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "MockProvider",
    "run_comparison_experiment",
]
