"""
Schema Corruptor: Synthetic data generation for training SchemaGNN.

Creates pairs of (corrupted_schema, error_labels) by systematically
breaking valid JSON schemas in detectable ways.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


class CorruptionType(Enum):
    """Types of corruptions that can be applied to schemas."""
    DANGLING_REF = auto()       # $ref points to non-existent definition
    CIRCULAR_REF = auto()       # Circular reference chain
    TYPE_MISMATCH = auto()      # Type changed without updating structure
    CONSTRAINT_CONFLICT = auto() # Conflicting constraints (min > max)
    MISSING_REQUIRED = auto()   # Required property not defined
    INVALID_PATTERN = auto()    # Invalid regex pattern
    WRONG_ITEMS_TYPE = auto()   # Array items has wrong type  
    ORPHAN_DEFINITION = auto()  # Definition never referenced (soft error)


@dataclass 
class CorruptionResult:
    """Result of applying a corruption to a schema."""
    corrupted_schema: dict
    corruption_type: CorruptionType
    corrupted_paths: list[str]  # JSON paths that were corrupted
    description: str            # Human-readable description


class SchemaCorruptor:
    """
    Generates corrupted JSON schemas for training the GNN.
    
    Applies systematic mutations to valid schemas to create
    training examples with known error locations.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the corruptor.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        # Register corruption strategies
        self._strategies: dict[CorruptionType, Callable] = {
            CorruptionType.DANGLING_REF: self._corrupt_dangling_ref,
            CorruptionType.CIRCULAR_REF: self._corrupt_circular_ref,
            CorruptionType.TYPE_MISMATCH: self._corrupt_type_mismatch,
            CorruptionType.CONSTRAINT_CONFLICT: self._corrupt_constraint_conflict,
            CorruptionType.MISSING_REQUIRED: self._corrupt_missing_required,
            CorruptionType.INVALID_PATTERN: self._corrupt_invalid_pattern,
            CorruptionType.WRONG_ITEMS_TYPE: self._corrupt_wrong_items_type,
        }
        
    def _find_refs(self, schema: dict, path: str = "") -> list[tuple[str, str]]:
        """Find all $ref locations in schema. Returns list of (path, ref_value)."""
        refs = []
        if isinstance(schema, dict):
            if "$ref" in schema:
                refs.append((path, schema["$ref"]))
            for key, value in schema.items():
                child_path = f"{path}.{key}" if path else key
                refs.extend(self._find_refs(value, child_path))
        elif isinstance(schema, list):
            for i, item in enumerate(schema):
                refs.extend(self._find_refs(item, f"{path}[{i}]"))
        return refs
        
    def _find_definitions(self, schema: dict) -> dict[str, str]:
        """Find all definitions and their paths."""
        defs = {}
        defs_key = "$defs" if "$defs" in schema else "definitions"
        if defs_key in schema:
            for name in schema[defs_key]:
                ref_path = f"#/{defs_key}/{name}"
                json_path = f"{defs_key}.{name}"
                defs[ref_path] = json_path
        return defs
        
    def _get_at_path(self, schema: dict, path: str) -> Any:
        """Get value at a JSON path."""
        if not path:
            return schema
        parts = path.replace("[", ".").replace("]", "").split(".")
        current = schema
        for part in parts:
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                return None
        return current
        
    def _set_at_path(self, schema: dict, path: str, value: Any) -> None:
        """Set value at a JSON path."""
        parts = path.replace("[", ".").replace("]", "").split(".")
        parts = [p for p in parts if p]
        current = schema
        for part in parts[:-1]:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = current[part]
        last_part = parts[-1]
        if last_part.isdigit():
            current[int(last_part)] = value
        else:
            current[last_part] = value
            
    def _find_properties(self, schema: dict, path: str = "") -> list[tuple[str, dict]]:
        """Find all property definitions. Returns list of (path, property_schema) where property_schema is a dict."""
        properties = []
        if isinstance(schema, dict):
            if "properties" in schema and isinstance(schema["properties"], dict):
                for prop_name, prop_schema in schema["properties"].items():
                    # Only include dict schemas (skip booleans and other primitives)
                    if isinstance(prop_schema, dict):
                        prop_path = f"{path}.properties.{prop_name}" if path else f"properties.{prop_name}"
                        properties.append((prop_path, prop_schema))
            for key, value in schema.items():
                if key not in ("properties",):
                    child_path = f"{path}.{key}" if path else key
                    properties.extend(self._find_properties(value, child_path))
        elif isinstance(schema, list):
            for i, item in enumerate(schema):
                properties.extend(self._find_properties(item, f"{path}[{i}]"))
        return properties
        
    def _find_arrays(self, schema: dict, path: str = "") -> list[tuple[str, dict]]:
        """Find all array type definitions."""
        arrays = []
        if isinstance(schema, dict):
            if schema.get("type") == "array":
                arrays.append((path, schema))
            for key, value in schema.items():
                child_path = f"{path}.{key}" if path else key
                arrays.extend(self._find_arrays(value, child_path))
        elif isinstance(schema, list):
            for i, item in enumerate(schema):
                arrays.extend(self._find_arrays(item, f"{path}[{i}]"))
        return arrays

    # === Corruption Strategies ===
    
    def _corrupt_dangling_ref(self, schema: dict) -> Optional[CorruptionResult]:
        """Create a $ref that points to a non-existent definition."""
        refs = self._find_refs(schema)
        if not refs:
            return None
            
        # Pick a random ref and break it
        path, original_ref = self.rng.choice(refs)
        fake_ref = f"#/definitions/NonExistent_{self.rng.randint(1000, 9999)}"
        
        corrupted = copy.deepcopy(schema)
        ref_obj = self._get_at_path(corrupted, path.rsplit(".", 1)[0]) if "." in path else corrupted
        if ref_obj and isinstance(ref_obj, dict):
            ref_obj["$ref"] = fake_ref
            
            return CorruptionResult(
                corrupted_schema=corrupted,
                corruption_type=CorruptionType.DANGLING_REF,
                corrupted_paths=[path],
                description=f"Changed $ref from '{original_ref}' to non-existent '{fake_ref}'"
            )
        return None
        
    def _corrupt_circular_ref(self, schema: dict) -> Optional[CorruptionResult]:
        """Create a circular reference loop."""
        defs = self._find_definitions(schema)
        if len(defs) < 2:
            return None
            
        corrupted = copy.deepcopy(schema)
        defs_key = "$defs" if "$defs" in corrupted else "definitions"
        
        # Pick two definitions and make them reference each other
        def_names = list(corrupted.get(defs_key, {}).keys())
        if len(def_names) < 2:
            return None
            
        def_a, def_b = self.rng.sample(def_names, 2)
        
        # Make def_a reference def_b
        corrupted[defs_key][def_a]["properties"] = corrupted[defs_key][def_a].get("properties", {})
        corrupted[defs_key][def_a]["properties"]["_circular"] = {"$ref": f"#/{defs_key}/{def_b}"}
        
        # Make def_b reference def_a (closing the loop)
        corrupted[defs_key][def_b]["properties"] = corrupted[defs_key][def_b].get("properties", {})
        corrupted[defs_key][def_b]["properties"]["_circular"] = {"$ref": f"#/{defs_key}/{def_a}"}
        
        return CorruptionResult(
            corrupted_schema=corrupted,
            corruption_type=CorruptionType.CIRCULAR_REF,
            corrupted_paths=[
                f"{defs_key}.{def_a}.properties._circular",
                f"{defs_key}.{def_b}.properties._circular"
            ],
            description=f"Created circular reference between '{def_a}' and '{def_b}'"
        )
        
    def _corrupt_type_mismatch(self, schema: dict) -> Optional[CorruptionResult]:
        """Change a type without updating the structure."""
        properties = self._find_properties(schema)
        if not properties:
            return None
            
        # Find an object property and change its type to string (filter out non-dict values)
        object_props = [(p, s) for p, s in properties if isinstance(s, dict) and s.get("type") == "object"]
        if not object_props:
            # Find any dict property and give it conflicting structure
            dict_props = [(p, s) for p, s in properties if isinstance(s, dict)]
            if not dict_props:
                return None
            path, prop_schema = self.rng.choice(dict_props)
        else:
            path, prop_schema = self.rng.choice(object_props)
            
        corrupted = copy.deepcopy(schema)
        corrupted_prop = self._get_at_path(corrupted, path)
        
        if corrupted_prop and isinstance(corrupted_prop, dict):
            # Change to string but keep object properties
            old_type = corrupted_prop.get("type", "object")
            corrupted_prop["type"] = "string"
            # Keep properties field which is invalid for strings
            if "properties" not in corrupted_prop:
                corrupted_prop["properties"] = {"invalid": {"type": "string"}}
                
            return CorruptionResult(
                corrupted_schema=corrupted,
                corruption_type=CorruptionType.TYPE_MISMATCH,
                corrupted_paths=[path],
                description=f"Changed type from '{old_type}' to 'string' but kept object properties"
            )
        return None
        
    def _corrupt_constraint_conflict(self, schema: dict) -> Optional[CorruptionResult]:
        """Create conflicting constraints (e.g., minItems > maxItems)."""
        corrupted = copy.deepcopy(schema)
        
        # Find a suitable location (array or number)
        arrays = self._find_arrays(corrupted)
        properties = self._find_properties(corrupted)
        
        if arrays:
            path, arr_schema = self.rng.choice(arrays)
            arr_obj = self._get_at_path(corrupted, path)
            if arr_obj:
                arr_obj["minItems"] = 10
                arr_obj["maxItems"] = 3
                
                return CorruptionResult(
                    corrupted_schema=corrupted,
                    corruption_type=CorruptionType.CONSTRAINT_CONFLICT,
                    corrupted_paths=[path],
                    description="Set minItems=10 but maxItems=3"
                )
                
        # Try with number constraints
        for path, prop in properties:
            prop_obj = self._get_at_path(corrupted, path)
            if isinstance(prop_obj, dict) and prop_obj.get("type") in ("number", "integer"):
                prop_obj["minimum"] = 100
                prop_obj["maximum"] = 10
                
                return CorruptionResult(
                    corrupted_schema=corrupted,
                    corruption_type=CorruptionType.CONSTRAINT_CONFLICT,
                    corrupted_paths=[path],
                    description="Set minimum=100 but maximum=10"
                )
                
        # Fallback: add constraint conflict to root
        corrupted["minProperties"] = 10
        corrupted["maxProperties"] = 2
        
        return CorruptionResult(
            corrupted_schema=corrupted,
            corruption_type=CorruptionType.CONSTRAINT_CONFLICT,
            corrupted_paths=["root"],
            description="Set minProperties=10 but maxProperties=2"
        )
        
    def _corrupt_missing_required(self, schema: dict) -> Optional[CorruptionResult]:
        """Add a required property that doesn't exist in properties."""
        corrupted = copy.deepcopy(schema)
        
        # Find an object with properties
        def find_objects_with_props(s: dict, path: str = "") -> list[tuple[str, dict]]:
            results = []
            if isinstance(s, dict):
                if "properties" in s:
                    results.append((path, s))
                for k, v in s.items():
                    child_path = f"{path}.{k}" if path else k
                    results.extend(find_objects_with_props(v, child_path))
            return results
            
        objects = find_objects_with_props(corrupted)
        if not objects:
            # Add to root
            corrupted["properties"] = {"existing": {"type": "string"}}
            corrupted["required"] = ["existing", "nonExistent123"]
            return CorruptionResult(
                corrupted_schema=corrupted,
                corruption_type=CorruptionType.MISSING_REQUIRED,
                corrupted_paths=["root.required"],
                description="Required 'nonExistent123' but not defined in properties"
            )
            
        path, obj = self.rng.choice(objects)
        fake_required = f"missingProp_{self.rng.randint(1000, 9999)}"
        
        existing_required = list(obj.get("required", []))
        existing_required.append(fake_required)
        obj["required"] = existing_required
        
        return CorruptionResult(
            corrupted_schema=corrupted,
            corruption_type=CorruptionType.MISSING_REQUIRED,
            corrupted_paths=[f"{path}.required" if path else "required"],
            description=f"Required '{fake_required}' but not defined in properties"
        )
        
    def _corrupt_invalid_pattern(self, schema: dict) -> Optional[CorruptionResult]:
        """Add an invalid regex pattern."""
        corrupted = copy.deepcopy(schema)
        properties = self._find_properties(corrupted)
        
        # Find a string property (filter out non-dict values like booleans)
        string_props = [(p, s) for p, s in properties 
                       if isinstance(s, dict) and (s.get("type") == "string" or "pattern" in s)]
        
        if string_props:
            path, _ = self.rng.choice(string_props)
        else:
            # Create one
            if "properties" not in corrupted:
                corrupted["properties"] = {}
            corrupted["properties"]["_patternTest"] = {"type": "string"}
            path = "properties._patternTest"
            
        prop_obj = self._get_at_path(corrupted, path)
        if prop_obj:
            prop_obj["pattern"] = "[invalid(regex"  # Unclosed bracket
            
            return CorruptionResult(
                corrupted_schema=corrupted,
                corruption_type=CorruptionType.INVALID_PATTERN,
                corrupted_paths=[path],
                description="Added invalid regex pattern '[invalid(regex'"
            )
        return None
        
    def _corrupt_wrong_items_type(self, schema: dict) -> Optional[CorruptionResult]:
        """Make array items have wrong type configuration."""
        corrupted = copy.deepcopy(schema)
        arrays = self._find_arrays(corrupted)
        
        if not arrays:
            # Create an array
            if "properties" not in corrupted:
                corrupted["properties"] = {}
            corrupted["properties"]["_arrayTest"] = {"type": "array", "items": {"type": "object"}}
            arrays = [("properties._arrayTest", corrupted["properties"]["_arrayTest"])]
            
        path, _ = self.rng.choice(arrays)
        arr_obj = self._get_at_path(corrupted, path)
        
        if arr_obj:
            # Give it a non-schema items value
            arr_obj["items"] = "invalid_not_a_schema"
            
            return CorruptionResult(
                corrupted_schema=corrupted,
                corruption_type=CorruptionType.WRONG_ITEMS_TYPE,
                corrupted_paths=[f"{path}.items"],
                description="Set array items to a string instead of a schema"
            )
        return None
        
    def corrupt(
        self,
        schema: dict,
        corruption_type: Optional[CorruptionType] = None,
        num_corruptions: int = 1,
    ) -> list[CorruptionResult]:
        """
        Apply corruption(s) to a schema.
        
        Args:
            schema: Valid JSON schema to corrupt
            corruption_type: Specific corruption to apply, or None for random
            num_corruptions: Number of corruptions to apply
            
        Returns:
            List of CorruptionResult objects
        """
        results = []
        available_types = list(self._strategies.keys())
        
        for _ in range(num_corruptions):
            if corruption_type:
                ct = corruption_type
            else:
                ct = self.rng.choice(available_types)
                
            strategy = self._strategies[ct]
            result = strategy(schema)
            
            if result:
                results.append(result)
                schema = result.corrupted_schema  # Chain corruptions
                
        return results
        
    def generate_training_pair(
        self,
        schema: dict,
        corruption_types: Optional[list[CorruptionType]] = None,
    ) -> dict:
        """
        Generate a training example from a valid schema.
        
        Args:
            schema: Valid JSON schema
            corruption_types: List of corruption types to choose from
            
        Returns:
            Dict with 'schema', 'is_valid', 'corrupted_paths', 'corruption_type'
        """
        if corruption_types is None:
            corruption_types = list(self._strategies.keys())
            
        ct = self.rng.choice(corruption_types)
        results = self.corrupt(schema, ct, num_corruptions=1)
        
        if results:
            result = results[0]
            return {
                "schema": result.corrupted_schema,
                "is_valid": False,
                "corrupted_paths": result.corrupted_paths,
                "corruption_type": result.corruption_type.name,
                "description": result.description,
            }
        else:
            # Corruption failed, return original as valid
            return {
                "schema": schema,
                "is_valid": True,
                "corrupted_paths": [],
                "corruption_type": None,
                "description": "Original valid schema",
            }
            
    def generate_dataset(
        self,
        schemas: list[dict],
        valid_ratio: float = 0.3,
        corruptions_per_schema: int = 3,
        seed: Optional[int] = None,
    ) -> list[dict]:
        """
        Generate a training dataset from a list of valid schemas.
        
        Args:
            schemas: List of valid JSON schemas
            valid_ratio: Ratio of valid examples to include
            corruptions_per_schema: Number of corrupted versions per schema
            seed: Random seed
            
        Returns:
            List of training examples
        """
        if seed is not None:
            self.rng = random.Random(seed)
            
        dataset = []
        
        for schema in schemas:
            # Add valid version with probability valid_ratio
            if self.rng.random() < valid_ratio:
                dataset.append({
                    "schema": copy.deepcopy(schema),
                    "is_valid": True,
                    "corrupted_paths": [],
                    "corruption_type": None,
                    "description": "Original valid schema",
                })
                
            # Generate corrupted versions
            for _ in range(corruptions_per_schema):
                example = self.generate_training_pair(copy.deepcopy(schema))
                dataset.append(example)
                
        self.rng.shuffle(dataset)
        return dataset
