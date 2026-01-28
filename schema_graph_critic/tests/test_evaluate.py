"""Tests for the evaluation module."""

import pytest
import torch

from schema_graph_critic.evaluate import (
    ClassificationMetrics,
    EvaluationResult,
    Evaluator,
)
from schema_graph_critic.model import SchemaGNN
from schema_graph_critic.parser import SchemaGraphParser


class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""
    
    def test_from_confusion_matrix_perfect(self):
        """Test metrics with perfect predictions."""
        metrics = ClassificationMetrics.from_confusion_matrix(
            tp=50, tn=50, fp=0, fn=0
        )
        
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        
    def test_from_confusion_matrix_all_wrong(self):
        """Test metrics with all wrong predictions."""
        metrics = ClassificationMetrics.from_confusion_matrix(
            tp=0, tn=0, fp=50, fn=50
        )
        
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        
    def test_from_confusion_matrix_partial(self):
        """Test metrics with partial correct predictions."""
        metrics = ClassificationMetrics.from_confusion_matrix(
            tp=40, tn=30, fp=10, fn=20
        )
        
        assert metrics.accuracy == 0.7  # 70/100
        assert metrics.precision == 0.8  # 40/50
        assert metrics.recall == pytest.approx(0.6666, rel=0.01)  # 40/60
        
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ClassificationMetrics.from_confusion_matrix(
            tp=10, tn=10, fp=5, fn=5
        )
        
        d = metrics.to_dict()
        
        assert "accuracy" in d
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert d["true_positives"] == 10
        assert d["false_positives"] == 5


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        global_metrics = ClassificationMetrics.from_confusion_matrix(
            tp=50, tn=50, fp=0, fn=0
        )
        node_metrics = ClassificationMetrics.from_confusion_matrix(
            tp=100, tn=200, fp=10, fn=20
        )
        
        result = EvaluationResult(
            global_metrics=global_metrics,
            node_metrics=node_metrics,
            num_samples=100,
            num_valid_samples=50,
            num_invalid_samples=50,
        )
        
        d = result.to_dict()
        
        assert "global_metrics" in d
        assert "node_metrics" in d
        assert d["num_samples"] == 100
        
    def test_summary(self):
        """Test summary generation."""
        global_metrics = ClassificationMetrics.from_confusion_matrix(
            tp=45, tn=45, fp=5, fn=5
        )
        node_metrics = ClassificationMetrics.from_confusion_matrix(
            tp=100, tn=200, fp=10, fn=20
        )
        
        result = EvaluationResult(
            global_metrics=global_metrics,
            node_metrics=node_metrics,
            num_samples=100,
            num_valid_samples=50,
            num_invalid_samples=50,
        )
        
        summary = result.summary()
        
        assert "EVALUATION RESULTS" in summary
        assert "GLOBAL VALIDITY PREDICTION" in summary
        assert "NODE-LEVEL ERROR DETECTION" in summary
        assert "90.00%" in summary  # accuracy


class TestEvaluator:
    """Tests for Evaluator class."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        return SchemaGNN(
            input_dim=404,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
        )
        
    @pytest.fixture
    def parser(self):
        """Create a test parser."""
        return SchemaGraphParser(device="cpu")
        
    @pytest.fixture
    def evaluator(self, model, parser):
        """Create evaluator instance."""
        return Evaluator(
            model=model,
            parser=parser,
            device="cpu",
        )
        
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes correctly."""
        assert evaluator.model is not None
        assert evaluator.parser is not None
        assert evaluator.validity_threshold == 0.5
        assert evaluator.error_threshold == 0.5
        
    def test_evaluate_simple(self, evaluator):
        """Test evaluation on simple examples."""
        test_examples = [
            {
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                "is_valid": True,
                "corrupted_paths": [],
                "corruption_type": "NONE",
            },
            {
                "schema": {"type": "object", "properties": {"age": {"type": "integer", "minimum": 10, "maximum": 5}}},
                "is_valid": False,
                "corrupted_paths": ["properties.age"],
                "corruption_type": "CONSTRAINT_CONFLICT",
            },
        ]
        
        result = evaluator.evaluate(test_examples, show_progress=False)
        
        assert isinstance(result, EvaluationResult)
        assert result.num_samples == 2
        assert result.num_valid_samples == 1
        assert result.num_invalid_samples == 1
        
    def test_prepare_test_data(self, evaluator):
        """Test test data preparation from schemas."""
        schemas = [
            {"type": "object", "properties": {"x": {"type": "string"}}},
            {"type": "array", "items": {"type": "integer"}},
        ]
        
        examples = evaluator._prepare_test_data(
            schemas,
            valid_ratio=0.5,
            corruptions_per_schema=1,
            seed=42,
        )
        
        assert len(examples) > 0
        assert all("schema" in ex and "is_valid" in ex for ex in examples)
