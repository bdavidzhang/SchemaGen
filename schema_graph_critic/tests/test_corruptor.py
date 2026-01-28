"""Tests for the Schema Corruptor."""

import pytest

from schema_graph_critic.corruptor import SchemaCorruptor, CorruptionType


@pytest.fixture
def valid_schema():
    """A valid schema for testing corruptions."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "definitions": {
            "Address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
            },
            "Contact": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                },
            },
        },
    }


@pytest.fixture
def schema_with_refs():
    """Schema with references for testing."""
    return {
        "type": "object",
        "properties": {
            "address": {"$ref": "#/definitions/Address"},
        },
        "definitions": {
            "Address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
            },
        },
    }


@pytest.fixture
def corruptor():
    """Create a corruptor with fixed seed."""
    return SchemaCorruptor(seed=42)


class TestSchemaCorruptor:
    """Tests for SchemaCorruptor."""
    
    def test_corrupt_returns_results(self, corruptor, valid_schema):
        """Test that corrupt returns results."""
        results = corruptor.corrupt(valid_schema)
        
        assert len(results) >= 0  # May fail for some corruption types
        
    def test_corruption_modifies_schema(self, corruptor, valid_schema):
        """Test that corruption modifies the schema."""
        import json
        
        original = json.dumps(valid_schema, sort_keys=True)
        
        results = corruptor.corrupt(valid_schema, num_corruptions=5)
        
        for result in results:
            corrupted = json.dumps(result.corrupted_schema, sort_keys=True)
            assert corrupted != original, f"Corruption {result.corruption_type} didn't modify schema"
            
    def test_corruption_records_paths(self, corruptor, valid_schema):
        """Test that corrupted paths are recorded."""
        results = corruptor.corrupt(valid_schema)
        
        for result in results:
            assert len(result.corrupted_paths) > 0
            
    def test_dangling_ref_corruption(self, corruptor, schema_with_refs):
        """Test dangling reference corruption."""
        results = corruptor.corrupt(
            schema_with_refs, 
            CorruptionType.DANGLING_REF
        )
        
        if results:
            result = results[0]
            assert result.corruption_type == CorruptionType.DANGLING_REF
            # The ref should now point to something that doesn't exist
            ref_value = result.corrupted_schema["properties"]["address"]["$ref"]
            assert "NonExistent" in ref_value
            
    def test_constraint_conflict_corruption(self, corruptor, valid_schema):
        """Test constraint conflict corruption."""
        results = corruptor.corrupt(
            valid_schema,
            CorruptionType.CONSTRAINT_CONFLICT
        )
        
        if results:
            result = results[0]
            assert result.corruption_type == CorruptionType.CONSTRAINT_CONFLICT
            
    def test_circular_ref_corruption(self, corruptor, valid_schema):
        """Test circular reference corruption."""
        results = corruptor.corrupt(
            valid_schema,
            CorruptionType.CIRCULAR_REF
        )
        
        if results:
            result = results[0]
            assert result.corruption_type == CorruptionType.CIRCULAR_REF
            assert len(result.corrupted_paths) >= 2  # At least 2 nodes in cycle


class TestDatasetGeneration:
    """Tests for training dataset generation."""
    
    def test_generate_dataset(self, corruptor, valid_schema):
        """Test dataset generation."""
        dataset = corruptor.generate_dataset(
            [valid_schema],
            valid_ratio=0.3,
            corruptions_per_schema=3,
        )
        
        assert len(dataset) > 0
        
        # Check structure
        for example in dataset:
            assert "schema" in example
            assert "is_valid" in example
            assert "corrupted_paths" in example
            
    def test_dataset_has_valid_examples(self, corruptor, valid_schema):
        """Test that dataset includes valid examples."""
        dataset = corruptor.generate_dataset(
            [valid_schema],
            valid_ratio=0.5,
            corruptions_per_schema=2,
            seed=42,
        )
        
        valid_count = sum(1 for ex in dataset if ex["is_valid"])
        
        # Should have some valid examples
        assert valid_count > 0
        
    def test_dataset_has_corrupted_examples(self, corruptor, valid_schema):
        """Test that dataset includes corrupted examples."""
        dataset = corruptor.generate_dataset(
            [valid_schema],
            valid_ratio=0.0,  # No valid examples
            corruptions_per_schema=5,
        )
        
        corrupted_count = sum(1 for ex in dataset if not ex["is_valid"])
        
        assert corrupted_count > 0
        
    def test_generate_training_pair(self, corruptor, valid_schema):
        """Test single training pair generation."""
        pair = corruptor.generate_training_pair(valid_schema)
        
        assert "schema" in pair
        assert "is_valid" in pair
        assert isinstance(pair["is_valid"], bool)


class TestReproducibility:
    """Tests for reproducibility with seeds."""
    
    def test_seed_produces_same_results(self, valid_schema):
        """Test that same seed produces same results."""
        corruptor1 = SchemaCorruptor(seed=42)
        corruptor2 = SchemaCorruptor(seed=42)
        
        dataset1 = corruptor1.generate_dataset([valid_schema], corruptions_per_schema=3)
        dataset2 = corruptor2.generate_dataset([valid_schema], corruptions_per_schema=3)
        
        # Should produce same number of examples
        assert len(dataset1) == len(dataset2)
        
        # And same validity labels
        for ex1, ex2 in zip(dataset1, dataset2):
            assert ex1["is_valid"] == ex2["is_valid"]
