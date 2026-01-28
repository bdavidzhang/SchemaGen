"""
Deterministic Baselines for JSON Schema Validation.

These baselines implement classic graph algorithms for detecting structural issues
that can be found without machine learning. They serve as comparison points for
the SchemaGNN to demonstrate the value of the learned approach.

Baselines implemented:
1. ReferenceChecker: Detects dangling $ref pointers
2. CycleDetector: Detects circular references
3. ConstraintChecker: Detects constraint conflicts (min > max, etc.)
4. StandardValidator: Wrapper around jsonschema for syntax validation
5. CombinedBaseline: Combines all baselines
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

try:
    import jsonschema
    from jsonschema import Draft7Validator, RefResolver
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


@dataclass
class ValidationIssue:
    """A detected validation issue."""
    path: str
    issue_type: str
    message: str
    severity: str = "error"  # error, warning


@dataclass  
class BaselineResult:
    """Result from a baseline validator."""
    is_valid: bool
    issues: list[ValidationIssue]
    error_paths: list[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": [
                {"path": i.path, "type": i.issue_type, "message": i.message, "severity": i.severity}
                for i in self.issues
            ],
            "error_paths": self.error_paths,
        }


class BaselineValidator(ABC):
    """Abstract base class for baseline validators."""
    
    @abstractmethod
    def validate(self, schema: dict) -> BaselineResult:
        """
        Validate a JSON schema.
        
        Args:
            schema: The JSON schema to validate
            
        Returns:
            BaselineResult with validation outcome
        """
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this baseline."""
        pass


class ReferenceChecker(BaselineValidator):
    """
    Detects dangling $ref pointers in JSON schemas.
    
    Uses depth-first traversal to find all $ref occurrences and verifies
    that each referenced definition exists.
    """
    
    @property
    def name(self) -> str:
        return "ReferenceChecker"
        
    def _collect_definitions(self, schema: dict) -> set[str]:
        """Collect all defined definition paths."""
        definitions = set()
        
        # Check both "definitions" and "$defs" (draft-2019+)
        for defs_key in ["definitions", "$defs"]:
            if defs_key in schema and isinstance(schema[defs_key], dict):
                for name in schema[defs_key].keys():
                    definitions.add(f"#/{defs_key}/{name}")
                    
        return definitions
        
    def _find_refs(self, obj: Any, path: str = "") -> list[tuple[str, str]]:
        """
        Recursively find all $ref values in the schema.
        
        Returns:
            List of (path_to_ref, ref_target) tuples
        """
        refs = []
        
        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                refs.append((path or "root", obj["$ref"]))
            for key, value in obj.items():
                if key != "$ref":
                    child_path = f"{path}.{key}" if path else key
                    refs.extend(self._find_refs(value, child_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                child_path = f"{path}[{i}]"
                refs.extend(self._find_refs(item, child_path))
                
        return refs
        
    def validate(self, schema: dict) -> BaselineResult:
        """Check for dangling references."""
        issues = []
        error_paths = []
        
        definitions = self._collect_definitions(schema)
        refs = self._find_refs(schema)
        
        for ref_path, ref_target in refs:
            # Skip if ref_target is not a string
            if not isinstance(ref_target, str):
                continue
                
            # Skip external references (URLs)
            if ref_target.startswith("http://") or ref_target.startswith("https://"):
                continue
                
            # Check if internal reference exists
            if ref_target.startswith("#/"):
                if ref_target not in definitions:
                    issues.append(ValidationIssue(
                        path=ref_path,
                        issue_type="DANGLING_REF",
                        message=f"Reference '{ref_target}' points to non-existent definition",
                    ))
                    error_paths.append(ref_path)
                    
        return BaselineResult(
            is_valid=len(issues) == 0,
            issues=issues,
            error_paths=error_paths,
        )


class CycleDetector(BaselineValidator):
    """
    Detects circular references in JSON schemas.
    
    Uses Tarjan's algorithm variant to detect cycles in the reference graph.
    """
    
    @property
    def name(self) -> str:
        return "CycleDetector"
        
    def _build_ref_graph(self, schema: dict) -> dict[str, list[str]]:
        """
        Build a graph of definitions and their references.
        
        Returns:
            Dict mapping definition name to list of referenced definitions
        """
        graph: dict[str, list[str]] = {}
        
        for defs_key in ["definitions", "$defs"]:
            if defs_key not in schema:
                continue
                
            defs = schema[defs_key]
            if not isinstance(defs, dict):
                continue
                
            for name, definition in defs.items():
                node = f"#/{defs_key}/{name}"
                graph[node] = []
                
                # Find refs in this definition
                refs = self._collect_refs(definition)
                for ref in refs:
                    if ref.startswith("#/"):
                        graph[node].append(ref)
                        
        return graph
        
    def _collect_refs(self, obj: Any) -> list[str]:
        """Collect all $ref values in an object."""
        refs = []
        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                refs.append(obj["$ref"])
            for value in obj.values():
                refs.extend(self._collect_refs(value))
        elif isinstance(obj, list):
            for item in obj:
                refs.extend(self._collect_refs(item))
        return refs
        
    def _find_cycles(self, graph: dict[str, list[str]]) -> list[list[str]]:
        """Find all cycles in the reference graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if neighbor in graph:  # Only traverse if it's a definition
                        dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    
            path.pop()
            rec_stack.remove(node)
            
        for node in graph:
            if node not in visited:
                dfs(node)
                
        return cycles
        
    def validate(self, schema: dict) -> BaselineResult:
        """Check for circular references."""
        issues = []
        error_paths = []
        
        graph = self._build_ref_graph(schema)
        cycles = self._find_cycles(graph)
        
        for cycle in cycles:
            cycle_str = " -> ".join(cycle)
            issues.append(ValidationIssue(
                path=cycle[0],
                issue_type="CIRCULAR_REF",
                message=f"Circular reference detected: {cycle_str}",
            ))
            error_paths.extend(cycle[:-1])  # All nodes in cycle except duplicate
            
        return BaselineResult(
            is_valid=len(issues) == 0,
            issues=issues,
            error_paths=list(set(error_paths)),
        )


class ConstraintChecker(BaselineValidator):
    """
    Detects constraint conflicts in JSON schemas.
    
    Checks for:
    - minimum > maximum
    - minLength > maxLength
    - minItems > maxItems
    - minProperties > maxProperties
    - exclusiveMinimum >= exclusiveMaximum
    - Invalid regex patterns
    """
    
    @property
    def name(self) -> str:
        return "ConstraintChecker"
        
    CONSTRAINT_PAIRS = [
        ("minimum", "maximum"),
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
        ("minProperties", "maxProperties"),
    ]
    
    EXCLUSIVE_PAIRS = [
        ("exclusiveMinimum", "exclusiveMaximum"),
    ]
    
    def _check_constraints(
        self,
        obj: Any,
        path: str = "",
    ) -> list[ValidationIssue]:
        """Recursively check for constraint conflicts."""
        issues = []
        
        if not isinstance(obj, dict):
            return issues
            
        current_path = path or "root"
        
        # Check min/max pairs
        for min_key, max_key in self.CONSTRAINT_PAIRS:
            if min_key in obj and max_key in obj:
                try:
                    min_val = float(obj[min_key])
                    max_val = float(obj[max_key])
                    if min_val > max_val:
                        issues.append(ValidationIssue(
                            path=current_path,
                            issue_type="CONSTRAINT_CONFLICT",
                            message=f"{min_key}={min_val} > {max_key}={max_val}",
                        ))
                except (TypeError, ValueError):
                    pass
                    
        # Check exclusive pairs
        for min_key, max_key in self.EXCLUSIVE_PAIRS:
            if min_key in obj and max_key in obj:
                try:
                    min_val = float(obj[min_key])
                    max_val = float(obj[max_key])
                    if min_val >= max_val:
                        issues.append(ValidationIssue(
                            path=current_path,
                            issue_type="CONSTRAINT_CONFLICT",
                            message=f"{min_key}={min_val} >= {max_key}={max_val}",
                        ))
                except (TypeError, ValueError):
                    pass
                    
        # Check pattern validity
        if "pattern" in obj and isinstance(obj["pattern"], str):
            try:
                re.compile(obj["pattern"])
            except re.error as e:
                issues.append(ValidationIssue(
                    path=current_path,
                    issue_type="INVALID_PATTERN",
                    message=f"Invalid regex pattern: {e}",
                ))
                
        # Recurse into nested objects
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                issues.extend(self._check_constraints(value, child_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    issues.extend(self._check_constraints(item, f"{child_path}[{i}]"))
                    
        return issues
        
    def validate(self, schema: dict) -> BaselineResult:
        """Check for constraint conflicts."""
        issues = self._check_constraints(schema)
        error_paths = [i.path for i in issues]
        
        return BaselineResult(
            is_valid=len(issues) == 0,
            issues=issues,
            error_paths=error_paths,
        )


class StandardValidator(BaselineValidator):
    """
    Wrapper around jsonschema library for standard validation.
    
    This validates that the schema is a valid JSON Schema according to the spec.
    """
    
    @property
    def name(self) -> str:
        return "StandardValidator"
        
    def validate(self, schema: dict) -> BaselineResult:
        """Validate using jsonschema library."""
        if not HAS_JSONSCHEMA:
            return BaselineResult(
                is_valid=True,
                issues=[ValidationIssue(
                    path="",
                    issue_type="DEPENDENCY_MISSING",
                    message="jsonschema library not installed",
                    severity="warning",
                )],
                error_paths=[],
            )
            
        issues = []
        error_paths = []
        
        try:
            # Check if schema is valid according to JSON Schema spec
            Draft7Validator.check_schema(schema)
        except jsonschema.SchemaError as e:
            issues.append(ValidationIssue(
                path=".".join(str(p) for p in e.absolute_path) or "root",
                issue_type="SCHEMA_ERROR",
                message=str(e.message),
            ))
            error_paths.append(
                ".".join(str(p) for p in e.absolute_path) or "root"
            )
        except Exception as e:
            issues.append(ValidationIssue(
                path="root",
                issue_type="VALIDATION_ERROR",
                message=str(e),
            ))
            error_paths.append("root")
            
        return BaselineResult(
            is_valid=len(issues) == 0,
            issues=issues,
            error_paths=error_paths,
        )


class CombinedBaseline(BaselineValidator):
    """
    Combines multiple baseline validators.
    
    Runs all validators and aggregates results.
    """
    
    def __init__(
        self,
        include_standard: bool = True,
        include_references: bool = True,
        include_cycles: bool = True,
        include_constraints: bool = True,
    ):
        """
        Initialize with selected validators.
        
        Args:
            include_standard: Include jsonschema validation
            include_references: Include dangling reference check
            include_cycles: Include cycle detection
            include_constraints: Include constraint conflict check
        """
        self.validators: list[BaselineValidator] = []
        
        if include_standard:
            self.validators.append(StandardValidator())
        if include_references:
            self.validators.append(ReferenceChecker())
        if include_cycles:
            self.validators.append(CycleDetector())
        if include_constraints:
            self.validators.append(ConstraintChecker())
            
    @property
    def name(self) -> str:
        validator_names = [v.name for v in self.validators]
        return f"CombinedBaseline({', '.join(validator_names)})"
        
    def validate(self, schema: dict) -> BaselineResult:
        """Run all validators and combine results."""
        all_issues = []
        all_error_paths = set()
        
        for validator in self.validators:
            result = validator.validate(schema)
            all_issues.extend(result.issues)
            all_error_paths.update(result.error_paths)
            
        return BaselineResult(
            is_valid=len(all_issues) == 0,
            issues=all_issues,
            error_paths=list(all_error_paths),
        )
        
    def validate_batch(
        self,
        schemas: list[dict],
    ) -> list[BaselineResult]:
        """Validate multiple schemas."""
        return [self.validate(schema) for schema in schemas]


def run_baseline_evaluation(
    test_examples: list[dict],
    baseline: Optional[BaselineValidator] = None,
) -> dict:
    """
    Run baseline evaluation on test examples.
    
    Args:
        test_examples: List of dicts with 'schema', 'is_valid'
        baseline: Baseline validator (defaults to CombinedBaseline)
        
    Returns:
        Dict with metrics
    """
    if baseline is None:
        baseline = CombinedBaseline()
        
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for example in test_examples:
        result = baseline.validate(example["schema"])
        pred_valid = result.is_valid
        true_valid = example["is_valid"]
        
        # We're detecting invalid schemas (invalid = positive class)
        if not true_valid and not pred_valid:
            tp += 1
        elif true_valid and pred_valid:
            tn += 1
        elif true_valid and not pred_valid:
            fp += 1
        else:
            fn += 1
            
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        "baseline": baseline.name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }
