"""Tests for the baselines module."""

import pytest

from schema_graph_critic.baselines import (
    ReferenceChecker,
    CycleDetector,
    ConstraintChecker,
    StandardValidator,
    CombinedBaseline,
    BaselineResult,
    run_baseline_evaluation,
)


class TestReferenceChecker:
    """Tests for ReferenceChecker baseline."""
    
    @pytest.fixture
    def checker(self):
        return ReferenceChecker()
        
    def test_valid_schema_no_refs(self, checker):
        """Test schema without references."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            }
        }
        
        result = checker.validate(schema)
        
        assert result.is_valid
        assert len(result.issues) == 0
        
    def test_valid_schema_with_refs(self, checker):
        """Test schema with valid references."""
        schema = {
            "type": "object",
            "properties": {
                "user": {"$ref": "#/definitions/User"}
            },
            "definitions": {
                "User": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
        
        result = checker.validate(schema)
        
        assert result.is_valid
        assert len(result.issues) == 0
        
    def test_dangling_reference(self, checker):
        """Test schema with dangling reference."""
        schema = {
            "type": "object",
            "properties": {
                "user": {"$ref": "#/definitions/NonExistent"}
            },
            "definitions": {}
        }
        
        result = checker.validate(schema)
        
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].issue_type == "DANGLING_REF"
        
    def test_multiple_dangling_refs(self, checker):
        """Test schema with multiple dangling references."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"$ref": "#/definitions/Missing1"},
                "b": {"$ref": "#/definitions/Missing2"},
            },
            "definitions": {}
        }
        
        result = checker.validate(schema)
        
        assert not result.is_valid
        assert len(result.issues) == 2
        
    def test_external_refs_ignored(self, checker):
        """Test that external (URL) references are ignored."""
        schema = {
            "type": "object",
            "properties": {
                "external": {"$ref": "https://example.com/schema.json"}
            }
        }
        
        result = checker.validate(schema)
        
        assert result.is_valid  # External refs are not checked
        
    def test_defs_keyword(self, checker):
        """Test $defs (draft 2019+) is also checked."""
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"}
            },
            "$defs": {
                "Item": {"type": "string"}
            }
        }
        
        result = checker.validate(schema)
        
        assert result.is_valid


class TestCycleDetector:
    """Tests for CycleDetector baseline."""
    
    @pytest.fixture
    def detector(self):
        return CycleDetector()
        
    def test_no_cycles(self, detector):
        """Test schema without cycles."""
        schema = {
            "definitions": {
                "A": {"type": "string"},
                "B": {"$ref": "#/definitions/A"},
            }
        }
        
        result = detector.validate(schema)
        
        assert result.is_valid
        
    def test_direct_cycle(self, detector):
        """Test schema with direct self-reference cycle."""
        schema = {
            "definitions": {
                "A": {"$ref": "#/definitions/A"}
            }
        }
        
        result = detector.validate(schema)
        
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].issue_type == "CIRCULAR_REF"
        
    def test_indirect_cycle(self, detector):
        """Test schema with indirect cycle (A -> B -> A)."""
        schema = {
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/A"},
            }
        }
        
        result = detector.validate(schema)
        
        assert not result.is_valid
        assert any(i.issue_type == "CIRCULAR_REF" for i in result.issues)
        
    def test_long_chain_no_cycle(self, detector):
        """Test long reference chain without cycle."""
        schema = {
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/C"},
                "C": {"$ref": "#/definitions/D"},
                "D": {"type": "string"},
            }
        }
        
        result = detector.validate(schema)
        
        assert result.is_valid


class TestConstraintChecker:
    """Tests for ConstraintChecker baseline."""
    
    @pytest.fixture
    def checker(self):
        return ConstraintChecker()
        
    def test_valid_constraints(self, checker):
        """Test schema with valid constraints."""
        schema = {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        }
        
        result = checker.validate(schema)
        
        assert result.is_valid
        
    def test_minimum_greater_than_maximum(self, checker):
        """Test constraint conflict: minimum > maximum."""
        schema = {
            "type": "integer",
            "minimum": 100,
            "maximum": 0,
        }
        
        result = checker.validate(schema)
        
        assert not result.is_valid
        assert result.issues[0].issue_type == "CONSTRAINT_CONFLICT"
        
    def test_minLength_greater_than_maxLength(self, checker):
        """Test constraint conflict: minLength > maxLength."""
        schema = {
            "type": "string",
            "minLength": 50,
            "maxLength": 10,
        }
        
        result = checker.validate(schema)
        
        assert not result.is_valid
        
    def test_minItems_greater_than_maxItems(self, checker):
        """Test constraint conflict: minItems > maxItems."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 10,
            "maxItems": 5,
        }
        
        result = checker.validate(schema)
        
        assert not result.is_valid
        
    def test_invalid_regex_pattern(self, checker):
        """Test invalid regex pattern."""
        schema = {
            "type": "string",
            "pattern": "[invalid(regex",
        }
        
        result = checker.validate(schema)
        
        assert not result.is_valid
        assert result.issues[0].issue_type == "INVALID_PATTERN"
        
    def test_nested_constraint_conflict(self, checker):
        """Test constraint conflict in nested property."""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "minimum": 150,
                    "maximum": 0,
                }
            }
        }
        
        result = checker.validate(schema)
        
        assert not result.is_valid


class TestCombinedBaseline:
    """Tests for CombinedBaseline."""
    
    def test_all_validators_run(self):
        """Test that combined baseline runs all validators."""
        baseline = CombinedBaseline(
            include_standard=False,  # Skip to avoid dependency issues
            include_references=True,
            include_cycles=True,
            include_constraints=True,
        )
        
        # Schema with multiple issues
        schema = {
            "type": "object",
            "properties": {
                "bad_ref": {"$ref": "#/definitions/Missing"},
                "bad_constraint": {"type": "integer", "minimum": 100, "maximum": 0},
            },
            "definitions": {}
        }
        
        result = baseline.validate(schema)
        
        assert not result.is_valid
        # Should have at least 2 issues (dangling ref + constraint conflict)
        assert len(result.issues) >= 2
        
    def test_valid_schema_passes_all(self):
        """Test that valid schema passes all validators."""
        baseline = CombinedBaseline(include_standard=False)
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
            }
        }
        
        result = baseline.validate(schema)
        
        assert result.is_valid


class TestBaselineEvaluation:
    """Tests for baseline evaluation function."""
    
    def test_run_baseline_evaluation(self):
        """Test running baseline evaluation on test examples."""
        test_examples = [
            {
                "schema": {"type": "string"},
                "is_valid": True,
            },
            {
                "schema": {"type": "integer", "minimum": 100, "maximum": 0},
                "is_valid": False,
            },
            {
                "schema": {"$ref": "#/definitions/Missing", "definitions": {}},
                "is_valid": False,
            },
        ]
        
        baseline = CombinedBaseline(include_standard=False)
        results = run_baseline_evaluation(test_examples, baseline)
        
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert results["true_positives"] + results["true_negatives"] + \
               results["false_positives"] + results["false_negatives"] == 3
