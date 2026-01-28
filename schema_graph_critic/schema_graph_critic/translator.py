"""
Feedback Translator: Converts GNN outputs to natural language for LLM refinement.

Translates tensor outputs (node IDs, probabilities) into actionable feedback
prompts that can be sent back to the LLM for schema correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch_geometric.data import HeteroData

from .model import SchemaGNN
from .parser import NodeType


@dataclass
class DefectReport:
    """Detailed report about a detected defect."""
    node_id: int
    json_path: str
    node_type: str
    probability: float
    likely_issue: str
    suggestion: str


@dataclass
class SchemaAnalysis:
    """Complete analysis result from the GNN."""
    is_valid: bool
    validity_score: float
    defects: list[DefectReport]
    feedback_prompt: str
    summary: str


class FeedbackTranslator:
    """
    Translates GNN outputs into natural language feedback for LLMs.
    
    The translator:
    1. Interprets model outputs (validity scores, node probabilities)
    2. Maps node IDs back to JSON paths
    3. Generates actionable feedback prompts
    """
    
    # Templates for different error types based on node characteristics
    ISSUE_TEMPLATES = {
        "REF": {
            "likely_issue": "This reference may point to a non-existent or incorrectly named definition.",
            "suggestion": "Verify that the $ref target exists in the definitions section and the path is correct.",
        },
        "OBJECT": {
            "likely_issue": "This object definition may have structural issues (missing properties, type conflicts).",
            "suggestion": "Check that all required properties are defined and types are consistent.",
        },
        "ARRAY": {
            "likely_issue": "This array definition may have invalid items schema or constraint conflicts.",
            "suggestion": "Verify the items schema is valid and constraints (minItems/maxItems) are consistent.",
        },
        "LOGIC": {
            "likely_issue": "This logical operator (anyOf/oneOf/allOf) may have inconsistent or invalid options.",
            "suggestion": "Ensure all options in the logical operator are valid schemas and don't conflict.",
        },
        "DEFINITION": {
            "likely_issue": "This definition may be circular, unused, or have internal inconsistencies.",
            "suggestion": "Check for circular references and ensure the definition is properly structured.",
        },
        "DEFAULT": {
            "likely_issue": "This schema element appears structurally inconsistent.",
            "suggestion": "Review the element's type, constraints, and relationships.",
        },
    }
    
    SEVERITY_THRESHOLDS = {
        "critical": 0.9,
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3,
    }
    
    def __init__(
        self,
        model: SchemaGNN,
        validity_threshold: float = 0.5,
        defect_threshold: float = 0.5,
        max_defects: int = 5,
    ):
        """
        Initialize the translator.
        
        Args:
            model: Trained SchemaGNN model
            validity_threshold: Threshold for considering a schema valid
            defect_threshold: Threshold for flagging a node as defective
            max_defects: Maximum number of defects to report
        """
        self.model = model
        self.validity_threshold = validity_threshold
        self.defect_threshold = defect_threshold
        self.max_defects = max_defects
        
    def _get_severity(self, probability: float) -> str:
        """Get severity level from probability."""
        for level, threshold in self.SEVERITY_THRESHOLDS.items():
            if probability >= threshold:
                return level
        return "info"
        
    def _get_issue_template(self, node_type: str) -> dict:
        """Get issue template for a node type."""
        return self.ISSUE_TEMPLATES.get(
            node_type, 
            self.ISSUE_TEMPLATES["DEFAULT"]
        )
        
    def _build_defect_report(
        self,
        node_id: int,
        json_path: str,
        node_type: str,
        probability: float,
    ) -> DefectReport:
        """Build a detailed defect report for a node."""
        template = self._get_issue_template(node_type)
        
        return DefectReport(
            node_id=node_id,
            json_path=json_path,
            node_type=node_type,
            probability=probability,
            likely_issue=template["likely_issue"],
            suggestion=template["suggestion"],
        )
        
    def _generate_feedback_prompt(
        self,
        validity_score: float,
        defects: list[DefectReport],
    ) -> str:
        """Generate a natural language feedback prompt for the LLM."""
        if not defects:
            if validity_score >= self.validity_threshold:
                return "The schema appears to be valid. No structural issues detected."
            else:
                return (
                    "The schema has a low validity score but no specific defects were identified. "
                    "Please review the overall structure for subtle inconsistencies."
                )
                
        # Build the prompt
        lines = [
            "## Schema Validation Failed",
            "",
            f"**Validity Score:** {validity_score:.1%}",
            f"**Issues Found:** {len(defects)}",
            "",
            "### Detected Problems:",
            "",
        ]
        
        for i, defect in enumerate(defects, 1):
            severity = self._get_severity(defect.probability)
            lines.extend([
                f"**{i}. `{defect.json_path}`** ({severity.upper()} - {defect.probability:.1%})",
                f"   - **Type:** {defect.node_type}",
                f"   - **Issue:** {defect.likely_issue}",
                f"   - **Fix:** {defect.suggestion}",
                "",
            ])
            
        lines.extend([
            "### Instructions:",
            "",
            "Please fix the issues listed above and regenerate the schema. Focus on:",
        ])
        
        # Add specific focus areas based on defect types
        focus_areas = set()
        for defect in defects:
            if defect.node_type == "REF":
                focus_areas.add("- Verify all $ref targets exist and paths are correct")
            elif defect.node_type == "ARRAY":
                focus_areas.add("- Check array item schemas and constraints")
            elif defect.node_type == "LOGIC":
                focus_areas.add("- Review anyOf/oneOf/allOf options for consistency")
            elif defect.node_type == "OBJECT":
                focus_areas.add("- Ensure object properties match their type declarations")
                
        if not focus_areas:
            focus_areas.add("- Review the flagged paths for structural issues")
            
        lines.extend(sorted(focus_areas))
        
        return "\n".join(lines)
        
    def _generate_summary(
        self,
        is_valid: bool,
        validity_score: float,
        num_defects: int,
    ) -> str:
        """Generate a brief summary of the analysis."""
        if is_valid:
            return f"Schema is valid (score: {validity_score:.1%})"
            
        severity = "critical" if num_defects >= 3 else "moderate" if num_defects >= 1 else "minor"
        return f"Schema has {severity} issues: {num_defects} defect(s) found (score: {validity_score:.1%})"
        
    @torch.no_grad()
    def analyze(self, data: HeteroData, use_defect_based_validation: bool = True) -> SchemaAnalysis:
        """
        Analyze a schema graph and generate feedback.
        
        
        Args:
            data: HeteroData object from SchemaGraphParser
            use_defect_based_validation: If True, consider schema valid if no defects
                are found, regardless of global validity score. This is more robust
                for complex schemas that the model hasn't seen during training.
            
        Returns:
            SchemaAnalysis with validity info, defects, and feedback prompt
        """
        self.model.eval()
        
        # Run inference
        output = self.model(data)
        
        validity_score = output["validity_score"].item()
        node_probs = output["node_error_probs"].cpu().numpy()
        
        # Find defective nodes first
        defect_indices = (node_probs >= self.defect_threshold).nonzero()[0]
        num_defects = len(defect_indices)
        
        # Determine validity - use defect-based validation for robustness
        if use_defect_based_validation:
            # If no defects found, consider valid (more robust for complex schemas)
            is_valid = num_defects == 0
        else:
            # Original behavior: use global validity score
            is_valid = validity_score >= self.validity_threshold
        
        # Build defect reports from previously identified defect indices
        defects = []
        
        # Sort by probability and take top-k
        sorted_indices = sorted(defect_indices, key=lambda i: node_probs[i], reverse=True)
        sorted_indices = sorted_indices[:self.max_defects]
        
        for idx in sorted_indices:
            # Bounds check to avoid IndexError
            idx = int(idx)
            if hasattr(data, "node_paths") and idx < len(data.node_paths):
                json_path = data.node_paths[idx]
            else:
                json_path = f"node_{idx}"
                
            if hasattr(data, "node_types") and idx < len(data.node_types):
                node_type = data.node_types[idx]
            else:
                node_type = "UNKNOWN"
            
            defect = self._build_defect_report(
                node_id=idx,
                json_path=json_path,
                node_type=node_type,
                probability=float(node_probs[idx]),
            )
            defects.append(defect)
            
        # Generate feedback
        feedback_prompt = self._generate_feedback_prompt(validity_score, defects)
        summary = self._generate_summary(is_valid, validity_score, len(defects))
        
        return SchemaAnalysis(
            is_valid=is_valid,
            validity_score=validity_score,
            defects=defects,
            feedback_prompt=feedback_prompt,
            summary=summary,
        )
        
    def format_for_llm(self, analysis: SchemaAnalysis) -> str:
        """
        Format analysis as a prompt to send back to an LLM.
        
        Args:
            analysis: SchemaAnalysis from analyze()
            
        Returns:
            Formatted string suitable for LLM input
        """
        if analysis.is_valid:
            return (
                "✓ Schema validation passed.\n"
                f"Validity score: {analysis.validity_score:.1%}\n"
                "No structural issues detected."
            )
            
        return (
            f"✗ Schema validation failed.\n\n"
            f"{analysis.feedback_prompt}"
        )
        
    def get_correction_prompt(
        self,
        analysis: SchemaAnalysis,
        original_schema: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate a complete prompt for asking an LLM to correct the schema.
        
        Args:
            analysis: SchemaAnalysis from analyze()
            original_schema: The original JSON schema as a string
            context: Optional context about what the schema should describe
            
        Returns:
            Complete prompt for schema correction
        """
        prompt_parts = [
            "# Schema Correction Request",
            "",
            "The following JSON Schema has structural issues that need to be fixed.",
            "",
        ]
        
        if context:
            prompt_parts.extend([
                "## Context",
                context,
                "",
            ])
            
        prompt_parts.extend([
            "## Current Schema (with issues)",
            "```json",
            original_schema,
            "```",
            "",
            analysis.feedback_prompt,
            "",
            "## Your Task",
            "",
            "Please provide a corrected version of the schema that fixes all identified issues.",
            "Return ONLY the corrected JSON Schema, properly formatted.",
        ])
        
        return "\n".join(prompt_parts)
