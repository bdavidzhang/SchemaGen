"""
Evaluation Module for SchemaGNN.

Provides comprehensive metrics for evaluating the model:
- Global validity prediction: Accuracy, Precision, Recall, F1
- Node-level error detection: Precision, Recall, F1
- Per-corruption-type performance breakdown
- Confusion matrices
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import HeteroData
from tqdm import tqdm

from .model import SchemaGNN
from .parser import SchemaGraphParser
from .corruptor import SchemaCorruptor, CorruptionType
from .trainer import SchemaDataset


@dataclass
class ClassificationMetrics:
    """Metrics for binary classification."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    @classmethod
    def from_confusion_matrix(
        cls,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
    ) -> "ClassificationMetrics":
        """Create metrics from confusion matrix components."""
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return cls(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
        )
        
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    global_metrics: ClassificationMetrics
    node_metrics: ClassificationMetrics
    per_corruption_metrics: dict[str, ClassificationMetrics] = field(default_factory=dict)
    num_samples: int = 0
    num_valid_samples: int = 0
    num_invalid_samples: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "global_metrics": self.global_metrics.to_dict(),
            "node_metrics": self.node_metrics.to_dict(),
            "num_samples": self.num_samples,
            "num_valid_samples": self.num_valid_samples,
            "num_invalid_samples": self.num_invalid_samples,
        }
        if self.per_corruption_metrics:
            result["per_corruption_metrics"] = {
                k: v.to_dict() for k, v in self.per_corruption_metrics.items()
            }
        return result
        
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            "",
            f"Total samples: {self.num_samples}",
            f"  Valid: {self.num_valid_samples} ({self.num_valid_samples/max(self.num_samples,1):.1%})",
            f"  Invalid: {self.num_invalid_samples} ({self.num_invalid_samples/max(self.num_samples,1):.1%})",
            "",
            "GLOBAL VALIDITY PREDICTION",
            "-" * 40,
            f"  Accuracy:  {self.global_metrics.accuracy:.2%}",
            f"  Precision: {self.global_metrics.precision:.2%}",
            f"  Recall:    {self.global_metrics.recall:.2%}",
            f"  F1 Score:  {self.global_metrics.f1:.2%}",
            "",
            "NODE-LEVEL ERROR DETECTION",
            "-" * 40,
            f"  Accuracy:  {self.node_metrics.accuracy:.2%}",
            f"  Precision: {self.node_metrics.precision:.2%}",
            f"  Recall:    {self.node_metrics.recall:.2%}",
            f"  F1 Score:  {self.node_metrics.f1:.2%}",
        ]
        
        if self.per_corruption_metrics:
            lines.extend([
                "",
                "PER-CORRUPTION-TYPE DETECTION (Invalid → Detected)",
                "-" * 40,
            ])
            for corruption_type, metrics in sorted(self.per_corruption_metrics.items()):
                lines.append(
                    f"  {corruption_type:25s} F1: {metrics.f1:.2%} "
                    f"(P: {metrics.precision:.2%}, R: {metrics.recall:.2%})"
                )
                
        lines.append("=" * 60)
        return "\n".join(lines)


class Evaluator:
    """
    Evaluator for SchemaGNN models.
    
    Computes comprehensive metrics on test data including:
    - Global schema validity classification
    - Node-level error localization
    - Per-corruption-type breakdown
    """
    
    def __init__(
        self,
        model: SchemaGNN,
        parser: SchemaGraphParser,
        device: str = "cpu",
        validity_threshold: float = 0.5,
        error_threshold: float = 0.5,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained SchemaGNN model
            parser: SchemaGraphParser for converting schemas to graphs
            device: Device to run evaluation on
            validity_threshold: Threshold for global validity prediction
            error_threshold: Threshold for node-level error detection
        """
        self.model = model.to(device)
        self.parser = parser
        self.device = device
        self.validity_threshold = validity_threshold
        self.error_threshold = error_threshold
        
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: str = "cpu",
        **kwargs,
    ) -> "Evaluator":
        """Load evaluator from a model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Get config from checkpoint
        config = checkpoint.get("config", {})
        hidden_dim = getattr(config, "hidden_dim", 256) if hasattr(config, "hidden_dim") else 256
        num_layers = getattr(config, "num_layers", 3) if hasattr(config, "num_layers") else 3
        num_heads = getattr(config, "num_heads", 4) if hasattr(config, "num_heads") else 4
        
        # Initialize model
        model = SchemaGNN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        parser = SchemaGraphParser(device=device)
        
        return cls(model, parser, device, **kwargs)
        
    def _prepare_test_data(
        self,
        schemas: list[dict],
        valid_ratio: float = 0.3,
        corruptions_per_schema: int = 3,
        seed: int = 123,
    ) -> list[dict]:
        """Generate test examples from schemas."""
        corruptor = SchemaCorruptor(seed=seed)
        return corruptor.generate_dataset(
            schemas,
            valid_ratio=valid_ratio,
            corruptions_per_schema=corruptions_per_schema,
        )
        
    @torch.no_grad()
    def evaluate(
        self,
        test_examples: list[dict],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate model on test examples.
        
        Args:
            test_examples: List of dicts with 'schema', 'is_valid', 'corrupted_paths', 'corruption_type'
            batch_size: Batch size (currently only 1 supported for HeteroData)
            show_progress: Show progress bar
            
        Returns:
            EvaluationResult with all metrics
        """
        self.model.eval()
        
        # Create dataset
        dataset = SchemaDataset(test_examples, self.parser)
        
        def collate_fn(batch):
            return batch
            
        loader = TorchDataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        # Tracking variables
        global_tp, global_tn, global_fp, global_fn = 0, 0, 0, 0
        node_tp, node_tn, node_fp, node_fn = 0, 0, 0, 0
        
        # Per-corruption tracking
        corruption_stats: dict[str, dict] = {}
        
        num_valid = 0
        num_invalid = 0
        
        iterator = tqdm(loader, desc="Evaluating") if show_progress else loader
        
        for i, batch in enumerate(iterator):
            for data in batch:
                data = data.to(self.device)
                example = test_examples[i]
                
                # Get predictions
                output = self.model(data)
                
                pred_valid = output["validity_score"].item() >= self.validity_threshold
                true_valid = data.y.item() >= 0.5
                
                # Track valid/invalid counts
                if true_valid:
                    num_valid += 1
                else:
                    num_invalid += 1
                    
                # Global metrics (predicting invalid as positive class)
                if not true_valid and not pred_valid:  # True negative (correctly identified invalid)
                    global_tp += 1
                elif true_valid and pred_valid:  # True positive (correctly identified valid)
                    global_tn += 1
                elif true_valid and not pred_valid:  # False negative (valid marked as invalid)
                    global_fp += 1
                else:  # False positive (invalid marked as valid)
                    global_fn += 1
                    
                # Node-level metrics
                pred_errors = (output["node_error_probs"] >= self.error_threshold).cpu()
                true_errors = data.node_y.bool().cpu()
                
                node_tp += ((pred_errors == True) & (true_errors == True)).sum().item()
                node_tn += ((pred_errors == False) & (true_errors == False)).sum().item()
                node_fp += ((pred_errors == True) & (true_errors == False)).sum().item()
                node_fn += ((pred_errors == False) & (true_errors == True)).sum().item()
                
                # Per-corruption-type tracking
                corruption_type = example.get("corruption_type", "UNKNOWN")
                if corruption_type and corruption_type != "NONE":
                    if corruption_type not in corruption_stats:
                        corruption_stats[corruption_type] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
                        
                    stats = corruption_stats[corruption_type]
                    if not true_valid and not pred_valid:
                        stats["tp"] += 1
                    elif true_valid and pred_valid:
                        stats["tn"] += 1
                    elif true_valid and not pred_valid:
                        stats["fp"] += 1
                    else:
                        stats["fn"] += 1
                        
        # Compute final metrics
        global_metrics = ClassificationMetrics.from_confusion_matrix(
            global_tp, global_tn, global_fp, global_fn
        )
        node_metrics = ClassificationMetrics.from_confusion_matrix(
            node_tp, node_tn, node_fp, node_fn
        )
        
        per_corruption_metrics = {}
        for corruption_type, stats in corruption_stats.items():
            per_corruption_metrics[corruption_type] = ClassificationMetrics.from_confusion_matrix(
                stats["tp"], stats["tn"], stats["fp"], stats["fn"]
            )
            
        return EvaluationResult(
            global_metrics=global_metrics,
            node_metrics=node_metrics,
            per_corruption_metrics=per_corruption_metrics,
            num_samples=num_valid + num_invalid,
            num_valid_samples=num_valid,
            num_invalid_samples=num_invalid,
        )
        
    def evaluate_from_schemas(
        self,
        schemas: list[dict],
        valid_ratio: float = 0.3,
        corruptions_per_schema: int = 3,
        seed: int = 123,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate by generating test examples from valid schemas.
        
        Args:
            schemas: List of valid JSON schemas
            valid_ratio: Ratio of valid examples in test set
            corruptions_per_schema: Number of corruptions per schema
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to evaluate()
            
        Returns:
            EvaluationResult
        """
        test_examples = self._prepare_test_data(
            schemas,
            valid_ratio=valid_ratio,
            corruptions_per_schema=corruptions_per_schema,
            seed=seed,
        )
        return self.evaluate(test_examples, **kwargs)
        
    def compare_with_baseline(
        self,
        test_examples: list[dict],
        baseline_predictions: list[dict],
    ) -> dict:
        """
        Compare model performance with baseline predictions.
        
        Args:
            test_examples: Test examples with ground truth
            baseline_predictions: List of dicts with 'is_valid', 'error_paths' from baseline
            
        Returns:
            Dict with comparison metrics
        """
        # Get model predictions
        model_result = self.evaluate(test_examples, show_progress=False)
        
        # Compute baseline metrics
        baseline_tp, baseline_tn, baseline_fp, baseline_fn = 0, 0, 0, 0
        
        for example, pred in zip(test_examples, baseline_predictions):
            true_valid = example["is_valid"]
            pred_valid = pred["is_valid"]
            
            if not true_valid and not pred_valid:
                baseline_tp += 1
            elif true_valid and pred_valid:
                baseline_tn += 1
            elif true_valid and not pred_valid:
                baseline_fp += 1
            else:
                baseline_fn += 1
                
        baseline_metrics = ClassificationMetrics.from_confusion_matrix(
            baseline_tp, baseline_tn, baseline_fp, baseline_fn
        )
        
        return {
            "model": model_result.global_metrics.to_dict(),
            "baseline": baseline_metrics.to_dict(),
            "improvement": {
                "accuracy": model_result.global_metrics.accuracy - baseline_metrics.accuracy,
                "f1": model_result.global_metrics.f1 - baseline_metrics.f1,
                "precision": model_result.global_metrics.precision - baseline_metrics.precision,
                "recall": model_result.global_metrics.recall - baseline_metrics.recall,
            },
        }


def evaluate_checkpoint(
    checkpoint_path: Path,
    schema_dir: Path,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    device: str = "cpu",
) -> EvaluationResult:
    """
    Convenience function to evaluate a checkpoint on schemas.
    
    Args:
        checkpoint_path: Path to model checkpoint
        schema_dir: Directory containing JSON schema files
        output_path: Optional path to save results
        limit: Limit number of schemas to use
        device: Device to run on
        
    Returns:
        EvaluationResult
    """
    # Load schemas
    schema_files = list(schema_dir.glob("**/*.json"))
    if limit:
        schema_files = schema_files[:limit]
        
    schemas = []
    for path in schema_files:
        try:
            with open(path) as f:
                schemas.append(json.load(f))
        except Exception:
            continue
            
    # Create evaluator and run
    evaluator = Evaluator.from_checkpoint(checkpoint_path, device=device)
    result = evaluator.evaluate_from_schemas(schemas)
    
    # Print summary
    print(result.summary())
    
    # Save if requested
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
            
    return result
