"""
JSON Schema to Heterogeneous Graph Parser.

Converts JSON Schema documents into PyTorch Geometric HeteroData objects
for processing by the SchemaGNN model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer


class NodeType(Enum):
    """Types of nodes in the schema graph."""
    OBJECT = auto()
    ARRAY = auto()
    STRING = auto()
    NUMBER = auto()
    INTEGER = auto()
    BOOLEAN = auto()
    NULL = auto()
    REF = auto()
    DEFINITION = auto()
    LOGIC = auto()  # anyOf, oneOf, allOf
    ROOT = auto()


class EdgeType(Enum):
    """Types of edges (relationships) in the schema graph."""
    CONTAINS = "contains"        # Parent object → Child property
    ITEMS = "items"              # Array → Items definition
    REFERS_TO = "refers_to"      # $ref node → Target definition
    LOGIC = "logic"              # anyOf/oneOf → Options
    ADDITIONAL = "additional"    # additionalProperties edge


@dataclass
class SchemaNode:
    """Represents a node in the schema graph."""
    node_id: int
    node_type: NodeType
    json_path: str  # Path in original JSON (e.g., "definitions.user.address")
    depth: int
    title: str = ""
    description: str = ""
    constraints: dict = field(default_factory=dict)
    ref_target: Optional[str] = None  # For $ref nodes


@dataclass
class SchemaEdge:
    """Represents an edge in the schema graph."""
    source_id: int
    target_id: int
    edge_type: EdgeType
    property_name: str = ""  # For CONTAINS edges


class SchemaGraphParser:
    """
    Parses JSON Schema into a Heterogeneous Graph for GNN processing.
    
    The graph captures:
    - Node types: object, array, string, number, etc.
    - Edge types: contains, items, refers_to, logic
    - Semantic features: embeddings from schema descriptions
    - Structural features: depth, type one-hot, constraints
    """
    
    # Type string to NodeType mapping
    TYPE_MAP = {
        "object": NodeType.OBJECT,
        "array": NodeType.ARRAY,
        "string": NodeType.STRING,
        "number": NodeType.NUMBER,
        "integer": NodeType.INTEGER,
        "boolean": NodeType.BOOLEAN,
        "null": NodeType.NULL,
    }
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        device: str = "cpu"
    ):
        """
        Initialize the parser.
        
        Args:
            embedding_model: SentenceTransformer model for semantic embeddings
            embedding_dim: Dimension of the semantic embeddings
            device: Device to run embeddings on
        """
        self.embedding_dim = embedding_dim
        self.device = device
        self._encoder: Optional[SentenceTransformer] = None
        self._embedding_model_name = embedding_model
        
        # State for current parse
        self._nodes: list[SchemaNode] = []
        self._edges: list[SchemaEdge] = []
        self._node_counter: int = 0
        self._definitions_map: dict[str, int] = {}  # $ref path → node_id
        self._pending_refs: list[tuple[int, str]] = []  # (node_id, ref_path)
        
    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._embedding_model_name)
            self._encoder.to(self.device)
        return self._encoder
        
    def _reset_state(self) -> None:
        """Reset parser state for a new schema."""
        self._nodes = []
        self._edges = []
        self._node_counter = 0
        self._definitions_map = {}
        self._pending_refs = []
        
    def _create_node(
        self,
        node_type: NodeType,
        json_path: str,
        depth: int,
        schema_block: dict,
    ) -> int:
        """Create a new node and return its ID."""
        node_id = self._node_counter
        self._node_counter += 1
        
        node = SchemaNode(
            node_id=node_id,
            node_type=node_type,
            json_path=json_path,
            depth=depth,
            title=schema_block.get("title", ""),
            description=schema_block.get("description", ""),
            constraints=self._extract_constraints(schema_block),
            ref_target=schema_block.get("$ref"),
        )
        self._nodes.append(node)
        return node_id
        
    def _extract_constraints(self, schema: dict) -> dict:
        """Extract constraint fields from a schema block."""
        constraint_keys = {
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
            "minLength", "maxLength", "pattern", "format",
            "minItems", "maxItems", "uniqueItems",
            "minProperties", "maxProperties",
            "required", "enum", "const",
        }
        return {k: v for k, v in schema.items() if k in constraint_keys}
        
    def _add_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: EdgeType,
        property_name: str = ""
    ) -> None:
        """Add an edge to the graph."""
        self._edges.append(SchemaEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            property_name=property_name,
        ))
        
    def _parse_schema_block(
        self,
        schema: dict,
        json_path: str,
        depth: int,
        parent_id: Optional[int] = None,
        edge_type: Optional[EdgeType] = None,
        property_name: str = "",
    ) -> int:
        """
        Recursively parse a schema block into graph nodes and edges.
        
        Returns the node_id of the created node.
        """
        # Handle $ref
        if "$ref" in schema:
            node_id = self._create_node(NodeType.REF, json_path, depth, schema)
            ref_path = schema["$ref"]
            self._pending_refs.append((node_id, ref_path))
            
            if parent_id is not None and edge_type is not None:
                self._add_edge(parent_id, node_id, edge_type, property_name)
            return node_id
            
        # Handle logic operators (anyOf, oneOf, allOf)
        for logic_op in ["anyOf", "oneOf", "allOf"]:
            if logic_op in schema:
                node_id = self._create_node(NodeType.LOGIC, json_path, depth, schema)
                
                if parent_id is not None and edge_type is not None:
                    self._add_edge(parent_id, node_id, edge_type, property_name)
                    
                for i, option in enumerate(schema[logic_op]):
                    option_path = f"{json_path}.{logic_op}[{i}]"
                    self._parse_schema_block(
                        option, option_path, depth + 1,
                        parent_id=node_id,
                        edge_type=EdgeType.LOGIC,
                    )
                return node_id
                
        # Determine node type from 'type' field
        schema_type = schema.get("type", "object")
        if isinstance(schema_type, list):
            schema_type = schema_type[0]  # Take first type for simplicity
            
        node_type = self.TYPE_MAP.get(schema_type, NodeType.OBJECT)
        node_id = self._create_node(node_type, json_path, depth, schema)
        
        if parent_id is not None and edge_type is not None:
            self._add_edge(parent_id, node_id, edge_type, property_name)
            
        # Process properties (for objects)
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                prop_path = f"{json_path}.properties.{prop_name}"
                self._parse_schema_block(
                    prop_schema, prop_path, depth + 1,
                    parent_id=node_id,
                    edge_type=EdgeType.CONTAINS,
                    property_name=prop_name,
                )
                
        # Process additionalProperties
        if isinstance(schema.get("additionalProperties"), dict):
            add_path = f"{json_path}.additionalProperties"
            self._parse_schema_block(
                schema["additionalProperties"], add_path, depth + 1,
                parent_id=node_id,
                edge_type=EdgeType.ADDITIONAL,
            )
            
        # Process items (for arrays)
        if "items" in schema:
            items = schema["items"]
            if isinstance(items, dict):
                items_path = f"{json_path}.items"
                self._parse_schema_block(
                    items, items_path, depth + 1,
                    parent_id=node_id,
                    edge_type=EdgeType.ITEMS,
                )
            elif isinstance(items, list):
                for i, item_schema in enumerate(items):
                    item_path = f"{json_path}.items[{i}]"
                    self._parse_schema_block(
                        item_schema, item_path, depth + 1,
                        parent_id=node_id,
                        edge_type=EdgeType.ITEMS,
                    )
                    
        return node_id
        
    def _parse_definitions(self, schema: dict) -> None:
        """Parse the definitions/$defs section and build the reference map."""
        defs_key = "$defs" if "$defs" in schema else "definitions"
        if defs_key not in schema:
            return
            
        for def_name, def_schema in schema[defs_key].items():
            json_path = f"{defs_key}.{def_name}"
            ref_path = f"#/{defs_key}/{def_name}"
            
            node_id = self._create_node(
                NodeType.DEFINITION, json_path, depth=1, schema_block=def_schema
            )
            self._definitions_map[ref_path] = node_id
            
            # Parse the definition's contents
            self._parse_schema_block(
                def_schema,
                json_path,
                depth=2,
                parent_id=node_id,
                edge_type=EdgeType.CONTAINS,
            )
            
    def _resolve_refs(self) -> None:
        """Resolve $ref edges after all definitions are parsed."""
        for ref_node_id, ref_path in self._pending_refs:
            if ref_path in self._definitions_map:
                target_id = self._definitions_map[ref_path]
                self._add_edge(ref_node_id, target_id, EdgeType.REFERS_TO)
            else:
                # Mark as dangling reference (points to nothing)
                # This is an error the GNN should detect!
                self._nodes[ref_node_id].constraints["_dangling_ref"] = True
                
    def _compute_node_features(self) -> torch.Tensor:
        """
        Compute feature vectors for all nodes.
        
        Features:
        - Semantic embedding (384-dim): From title + description
        - Type one-hot (11-dim): Node type encoding
        - Depth (1-dim): Normalized nesting level
        - Constraint flags (8-dim): Has constraints encoding
        
        Total: 404 dimensions
        """
        num_nodes = len(self._nodes)
        num_types = len(NodeType)
        
        # Prepare text for semantic embeddings
        texts = []
        for node in self._nodes:
            text = f"{node.title} {node.description}".strip()
            if not text:
                text = node.json_path.split(".")[-1]  # Use property name
            texts.append(text)
            
        # Compute semantic embeddings
        with torch.no_grad():
            semantic_embeddings = self.encoder.encode(
                texts, convert_to_tensor=True, device=self.device
            )
            
        # Compute type one-hot
        type_one_hot = torch.zeros(num_nodes, num_types)
        for i, node in enumerate(self._nodes):
            type_one_hot[i, node.node_type.value - 1] = 1.0
            
        # Compute normalized depth
        max_depth = max(node.depth for node in self._nodes) or 1
        depths = torch.tensor(
            [[node.depth / max_depth] for node in self._nodes],
            dtype=torch.float32
        )
        
        # Compute constraint flags
        constraint_flags = torch.zeros(num_nodes, 8)
        constraint_categories = [
            ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"],
            ["minLength", "maxLength"],
            ["pattern", "format"],
            ["minItems", "maxItems", "uniqueItems"],
            ["minProperties", "maxProperties"],
            ["required"],
            ["enum", "const"],
            ["_dangling_ref"],  # Special flag for broken refs
        ]
        for i, node in enumerate(self._nodes):
            for j, keys in enumerate(constraint_categories):
                if any(k in node.constraints for k in keys):
                    constraint_flags[i, j] = 1.0
                    
        # Concatenate all features
        features = torch.cat([
            semantic_embeddings.cpu(),
            type_one_hot,
            depths,
            constraint_flags,
        ], dim=1)
        
        return features
        
    def _build_edge_index(self, edge_type: EdgeType) -> torch.Tensor:
        """Build edge index tensor for a specific edge type."""
        edges = [(e.source_id, e.target_id) 
                 for e in self._edges if e.edge_type == edge_type]
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
        
    def parse(self, schema: dict | str | Path) -> HeteroData:
        """
        Parse a JSON Schema into a PyTorch Geometric HeteroData object.
        
        Args:
            schema: JSON Schema as dict, JSON string, or path to file
            
        Returns:
            HeteroData object ready for GNN processing
        """
        # Load schema if needed
        if isinstance(schema, (str, Path)):
            path = Path(schema)
            if path.exists():
                with open(path) as f:
                    schema = json.load(f)
            else:
                schema = json.loads(str(schema))
                
        self._reset_state()
        
        # Parse definitions first (so refs can resolve)
        self._parse_definitions(schema)
        
        # Parse main schema
        root_id = self._parse_schema_block(schema, "root", depth=0)
        
        # Resolve all $ref edges
        self._resolve_refs()
        
        # Build HeteroData
        data = HeteroData()
        
        # Add node features (all nodes share same feature space)
        data["schema_node"].x = self._compute_node_features()
        data["schema_node"].num_nodes = len(self._nodes)
        
        # Add edges for each type
        for edge_type in EdgeType:
            edge_key = ("schema_node", edge_type.value, "schema_node")
            data[edge_key].edge_index = self._build_edge_index(edge_type)
            
        # Store metadata for feedback translation
        data.node_paths = [node.json_path for node in self._nodes]
        data.node_types = [node.node_type.name for node in self._nodes]
        data.definitions_map = self._definitions_map.copy()
        
        return data
        
    def parse_batch(self, schemas: list[dict]) -> list[HeteroData]:
        """Parse multiple schemas into a list of HeteroData objects."""
        return [self.parse(schema) for schema in schemas]


# Convenience function
def schema_to_graph(
    schema: dict | str | Path,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> HeteroData:
    """
    Convert a JSON Schema to a heterogeneous graph.
    
    Args:
        schema: JSON Schema as dict, JSON string, or file path
        embedding_model: SentenceTransformer model for text embeddings
        
    Returns:
        PyTorch Geometric HeteroData object
    """
    parser = SchemaGraphParser(embedding_model=embedding_model)
    return parser.parse(schema)
