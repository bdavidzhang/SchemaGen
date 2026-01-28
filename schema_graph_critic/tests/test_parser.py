"""Tests for the JSON Schema to Graph parser."""

import pytest
import torch

from schema_graph_critic.parser import SchemaGraphParser, NodeType, EdgeType


@pytest.fixture
def simple_schema():
    """A simple schema for testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
        },
        "required": ["name"],
    }


@pytest.fixture
def schema_with_refs():
    """Schema with definitions and references."""
    return {
        "type": "object",
        "properties": {
            "user": {"$ref": "#/definitions/User"},
        },
        "definitions": {
            "User": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                },
            },
        },
    }


@pytest.fixture
def schema_with_arrays():
    """Schema with array types."""
    return {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
            },
        },
    }


@pytest.fixture
def parser():
    """Create a parser instance."""
    return SchemaGraphParser()


class TestSchemaGraphParser:
    """Tests for SchemaGraphParser."""
    
    def test_parse_simple_schema(self, parser, simple_schema):
        """Test parsing a simple schema."""
        graph = parser.parse(simple_schema)
        
        # Should have nodes
        assert graph["schema_node"].num_nodes > 0
        
        # Should have feature matrix
        assert graph["schema_node"].x.shape[0] == graph["schema_node"].num_nodes
        assert graph["schema_node"].x.shape[1] == 404  # Expected feature dim
        
        # Should have node paths
        assert hasattr(graph, "node_paths")
        assert len(graph.node_paths) == graph["schema_node"].num_nodes
        
    def test_parse_schema_with_refs(self, parser, schema_with_refs):
        """Test parsing schema with $ref."""
        graph = parser.parse(schema_with_refs)
        
        # Should have REFERS_TO edges
        edge_key = ("schema_node", "refers_to", "schema_node")
        assert edge_key in graph
        # The $ref should create an edge
        assert graph[edge_key].edge_index.shape[1] >= 1
        
    def test_parse_schema_with_arrays(self, parser, schema_with_arrays):
        """Test parsing schema with arrays."""
        graph = parser.parse(schema_with_arrays)
        
        # Should have ITEMS edges
        edge_key = ("schema_node", "items", "schema_node")
        assert edge_key in graph
        
    def test_node_types_tracked(self, parser, simple_schema):
        """Test that node types are tracked."""
        graph = parser.parse(simple_schema)
        
        assert hasattr(graph, "node_types")
        assert len(graph.node_types) == graph["schema_node"].num_nodes
        
        # Should have object and string types
        assert "OBJECT" in graph.node_types or "STRING" in graph.node_types
        
    def test_contains_edges(self, parser, simple_schema):
        """Test that CONTAINS edges are created for properties."""
        graph = parser.parse(simple_schema)
        
        edge_key = ("schema_node", "contains", "schema_node")
        assert edge_key in graph
        # Root object should contain name and age
        assert graph[edge_key].edge_index.shape[1] >= 2
        
    def test_definitions_map(self, parser, schema_with_refs):
        """Test that definitions are mapped correctly."""
        graph = parser.parse(schema_with_refs)
        
        assert hasattr(graph, "definitions_map")
        assert "#/definitions/User" in graph.definitions_map


class TestNodeFeatures:
    """Tests for node feature computation."""
    
    def test_feature_dimensions(self, parser, simple_schema):
        """Test feature vector dimensions."""
        graph = parser.parse(simple_schema)
        
        features = graph["schema_node"].x
        
        # 384 semantic + 11 type + 1 depth + 8 constraints = 404
        assert features.shape[1] == 404
        
    def test_features_are_float(self, parser, simple_schema):
        """Test features are float tensors."""
        graph = parser.parse(simple_schema)
        
        assert graph["schema_node"].x.dtype == torch.float32


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_schema(self, parser):
        """Test parsing minimal schema."""
        graph = parser.parse({"type": "object"})
        
        assert graph["schema_node"].num_nodes >= 1
        
    def test_deeply_nested_schema(self, parser):
        """Test parsing deeply nested schema."""
        schema = {"type": "object", "properties": {}}
        current = schema["properties"]
        
        for i in range(10):
            current[f"level{i}"] = {"type": "object", "properties": {}}
            current = current[f"level{i}"]["properties"]
            
        graph = parser.parse(schema)
        
        # Should handle deep nesting
        assert graph["schema_node"].num_nodes >= 10
        
    def test_dangling_ref(self, parser):
        """Test schema with dangling reference."""
        schema = {
            "type": "object",
            "properties": {
                "broken": {"$ref": "#/definitions/NonExistent"},
            },
        }
        
        graph = parser.parse(schema)
        
        # Should still parse, but mark as dangling
        assert graph["schema_node"].num_nodes >= 1
