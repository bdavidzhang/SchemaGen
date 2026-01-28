"""
LLM Pipeline: End-to-end schema generation with GNN-based refinement.

This module implements the core refinement loop:
1. LLM generates initial schema from requirements
2. SchemaGNN validates and identifies defects
3. Feedback is translated to natural language
4. LLM refines based on feedback
5. Repeat until valid or max iterations reached

Supports multiple LLM providers (OpenAI, Anthropic, etc.)
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Callable

import torch
from torch_geometric.data import HeteroData

from .model import SchemaGNN
from .parser import SchemaGraphParser
from .translator import FeedbackTranslator, SchemaAnalysis


@dataclass
class GenerationResult:
    """Result from a single generation attempt."""
    schema: Optional[dict]
    raw_response: str
    parse_error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.schema is not None


@dataclass
class RefinementRound:
    """Data from a single refinement round."""
    round_num: int
    schema: Optional[dict]
    analysis: Optional[SchemaAnalysis]
    feedback_sent: Optional[str]
    duration_ms: float
    
    
@dataclass
class PipelineResult:
    """Complete result from the refinement pipeline."""
    original_requirements: str
    final_schema: Optional[dict]
    final_analysis: Optional[SchemaAnalysis]
    is_valid: bool
    rounds: list[RefinementRound] = field(default_factory=list)
    total_duration_ms: float = 0.0
    tokens_used: int = 0
    
    @property
    def num_rounds(self) -> int:
        return len(self.rounds)
        
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "original_requirements": self.original_requirements,
            "final_schema": self.final_schema,
            "is_valid": self.is_valid,
            "num_rounds": self.num_rounds,
            "total_duration_ms": self.total_duration_ms,
            "tokens_used": self.tokens_used,
            "rounds": [
                {
                    "round_num": r.round_num,
                    "schema": r.schema,
                    "is_valid": r.analysis.is_valid if r.analysis else None,
                    "validity_score": r.analysis.validity_score if r.analysis else None,
                    "num_defects": len(r.analysis.defects) if r.analysis else 0,
                    "duration_ms": r.duration_ms,
                }
                for r in self.rounds
            ],
        }
        
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "PIPELINE RESULT",
            "=" * 60,
            f"Valid: {'Yes' if self.is_valid else 'No'}",
            f"Rounds: {self.num_rounds}",
            f"Total time: {self.total_duration_ms:.0f}ms",
            "",
            "Round History:",
        ]
        
        for r in self.rounds:
            status = "✓ valid" if (r.analysis and r.analysis.is_valid) else "✗ invalid"
            score = f"score={r.analysis.validity_score:.1%}" if r.analysis else "no analysis"
            lines.append(f"  Round {r.round_num}: {status} ({score}, {r.duration_ms:.0f}ms)")
            
        return "\n".join(lines)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, tokens_used)
        """
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model name (e.g., gpt-4-turbo-preview, gpt-3.5-turbo)
            api_key: API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
        
    @property
    def name(self) -> str:
        return f"OpenAI/{self.model}"
        
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        """Generate using OpenAI API."""
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        
        return content, tokens


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            model: Model name
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        
    @property
    def name(self) -> str:
        return f"Anthropic/{self.model}"
        
    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        """Generate using Anthropic API."""
        client = self._get_client()
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = client.messages.create(**kwargs)
        
        content = response.content[0].text if response.content else ""
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return content, tokens


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Gemini provider.
        
        Args:
            model: Model name (e.g., gemini-1.5-pro, gemini-1.5-flash)
            api_key: API key (defaults to GOOGLE_API_KEY or GEMINI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._client = None
        
    @property
    def name(self) -> str:
        return f"Gemini/{self.model}"
        
    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        return self._client
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        """Generate using Gemini API."""
        client = self._get_client()
        
        # Combine system prompt with user prompt for Gemini
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        response = client.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        
        content = response.text if response.text else ""
        # Gemini doesn't provide token counts in the same way, estimate from text
        tokens = len(content.split()) + len(full_prompt.split())
        
        return content, tokens


class MockProvider(LLMProvider):
    """Mock provider for testing."""
    
    def __init__(self, responses: Optional[list[str]] = None):
        """
        Initialize mock provider.
        
        Args:
            responses: List of responses to return in order
        """
        self.responses = responses or []
        self._index = 0
        
    @property
    def name(self) -> str:
        return "Mock"
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        """Return next mock response."""
        if self._index < len(self.responses):
            response = self.responses[self._index]
            self._index += 1
            return response, len(response.split())
        return '{"type": "object", "properties": {}}', 10


class SchemaRefinementPipeline:
    """
    End-to-end schema generation and refinement pipeline.
    
    This pipeline:
    1. Takes natural language requirements
    2. Generates an initial JSON Schema via LLM
    3. Validates with SchemaGNN
    4. If invalid, sends feedback to LLM for refinement
    5. Repeats until valid or max iterations reached
    """
    
    SYSTEM_PROMPT = """You are an expert JSON Schema designer. Your task is to create 
valid, well-structured JSON Schemas based on requirements.

Guidelines:
- Use JSON Schema draft-07 or later
- Include appropriate constraints (type, format, required, etc.)
- Use $ref for reusable definitions when appropriate
- Ensure all references point to existing definitions
- Avoid circular references unless explicitly needed
- Use clear, descriptive property names

Return ONLY the JSON Schema, no explanations."""

    GENERATION_PROMPT_TEMPLATE = """Create a JSON Schema for the following requirements:

{requirements}

Return ONLY the valid JSON Schema."""

    REFINEMENT_PROMPT_TEMPLATE = """The JSON Schema you generated has issues that need to be fixed.

## Current Schema
```json
{schema}
```

## Validation Feedback
{feedback}

Please fix the issues and return the corrected JSON Schema. Return ONLY the JSON Schema."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: SchemaGNN,
        parser: SchemaGraphParser,
        max_rounds: int = 5,
        validity_threshold: float = 0.5,
        device: str = "cpu",
        temperature: float = 0.7,
    ):
        """
        Initialize the pipeline.
        
        Args:
            llm_provider: LLM provider for generation/refinement
            model: Trained SchemaGNN model
            parser: Schema graph parser
            max_rounds: Maximum refinement rounds
            validity_threshold: Threshold for considering schema valid
            device: Device for GNN inference
            temperature: LLM sampling temperature
        """
        self.llm = llm_provider
        self.model = model.to(device)
        self.parser = parser
        self.translator = FeedbackTranslator(
            model,
            validity_threshold=validity_threshold,
            defect_threshold=0.7,  # Higher threshold to reduce false positives
        )
        self.max_rounds = max_rounds
        self.validity_threshold = validity_threshold
        self.device = device
        self.temperature = temperature
        
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        llm_provider: LLMProvider,
        device: str = "cpu",
        **kwargs,
    ) -> "SchemaRefinementPipeline":
        """
        Create pipeline from a model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            llm_provider: LLM provider
            device: Device for inference
            **kwargs: Additional arguments for pipeline
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        config = checkpoint.get("config", {})
        hidden_dim = getattr(config, "hidden_dim", 256) if hasattr(config, "hidden_dim") else 256
        num_layers = getattr(config, "num_layers", 3) if hasattr(config, "num_layers") else 3
        num_heads = getattr(config, "num_heads", 4) if hasattr(config, "num_heads") else 4
        
        model = SchemaGNN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        parser = SchemaGraphParser(device=device)
        
        return cls(
            llm_provider=llm_provider,
            model=model,
            parser=parser,
            device=device,
            **kwargs,
        )
        
    def _extract_json(self, response: str) -> GenerationResult:
        """Extract JSON schema from LLM response."""
        # Try to find JSON in the response
        response = response.strip()
        
        # Remove markdown code fences if present
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
            
        response = response.strip()
        
        try:
            schema = json.loads(response)
            return GenerationResult(schema=schema, raw_response=response)
        except json.JSONDecodeError as e:
            return GenerationResult(
                schema=None,
                raw_response=response,
                parse_error=str(e),
            )
            
    def _analyze_schema(self, schema: dict) -> tuple[HeteroData, SchemaAnalysis]:
        """Parse and analyze a schema."""
        data = self.parser.parse(schema)
        data = data.to(self.device)
        analysis = self.translator.analyze(data)
        return data, analysis
        
    def generate(
        self,
        requirements: str,
        on_round_complete: Optional[Callable[[RefinementRound], None]] = None,
    ) -> PipelineResult:
        """
        Generate a JSON Schema from requirements with iterative refinement.
        
        Args:
            requirements: Natural language requirements
            on_round_complete: Optional callback after each round
            
        Returns:
            PipelineResult with final schema and round history
        """
        start_time = time.time()
        total_tokens = 0
        rounds: list[RefinementRound] = []
        
        current_schema: Optional[dict] = None
        current_analysis: Optional[SchemaAnalysis] = None
        
        for round_num in range(self.max_rounds):
            round_start = time.time()
            
            if round_num == 0:
                # Initial generation
                prompt = self.GENERATION_PROMPT_TEMPLATE.format(
                    requirements=requirements
                )
            else:
                # Refinement with feedback
                if current_analysis is None:
                    # Previous analysis failed, provide generic feedback
                    feedback = "The previous schema could not be analyzed. Please regenerate a valid JSON Schema."
                else:
                    feedback = self.translator.format_for_llm(current_analysis)
                prompt = self.REFINEMENT_PROMPT_TEMPLATE.format(
                    schema=json.dumps(current_schema, indent=2) if current_schema else "{}",
                    feedback=feedback,
                )
                
            # Generate from LLM
            response, tokens = self.llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=self.temperature,
            )
            total_tokens += tokens
            
            # Parse response
            result = self._extract_json(response)
            
            if not result.success:
                # JSON parse failed - record round and continue
                round_data = RefinementRound(
                    round_num=round_num,
                    schema=None,
                    analysis=None,
                    feedback_sent=f"Failed to parse JSON: {result.parse_error}",
                    duration_ms=(time.time() - round_start) * 1000,
                )
                rounds.append(round_data)
                
                if on_round_complete:
                    on_round_complete(round_data)
                continue
                
            current_schema = result.schema
            
            # Analyze with GNN
            try:
                _, current_analysis = self._analyze_schema(current_schema)
            except Exception as e:
                # Analysis failed - record and continue
                round_data = RefinementRound(
                    round_num=round_num,
                    schema=current_schema,
                    analysis=None,
                    feedback_sent=f"Analysis failed: {e}",
                    duration_ms=(time.time() - round_start) * 1000,
                )
                rounds.append(round_data)
                
                if on_round_complete:
                    on_round_complete(round_data)
                continue
                
            round_data = RefinementRound(
                round_num=round_num,
                schema=current_schema,
                analysis=current_analysis,
                feedback_sent=self.translator.format_for_llm(current_analysis) if not current_analysis.is_valid else None,
                duration_ms=(time.time() - round_start) * 1000,
            )
            rounds.append(round_data)
            
            if on_round_complete:
                on_round_complete(round_data)
                
            # Check if valid
            if current_analysis.is_valid:
                break
                
        total_duration = (time.time() - start_time) * 1000
        
        return PipelineResult(
            original_requirements=requirements,
            final_schema=current_schema,
            final_analysis=current_analysis,
            is_valid=current_analysis.is_valid if current_analysis else False,
            rounds=rounds,
            total_duration_ms=total_duration,
            tokens_used=total_tokens,
        )
        
    def generate_without_refinement(
        self,
        requirements: str,
    ) -> PipelineResult:
        """
        Generate schema without refinement (baseline comparison).
        
        Args:
            requirements: Natural language requirements
            
        Returns:
            PipelineResult from single generation
        """
        start_time = time.time()
        
        prompt = self.GENERATION_PROMPT_TEMPLATE.format(requirements=requirements)
        
        response, tokens = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
        )
        
        result = self._extract_json(response)
        
        analysis = None
        if result.success:
            try:
                _, analysis = self._analyze_schema(result.schema)
            except Exception:
                pass
                
        round_data = RefinementRound(
            round_num=0,
            schema=result.schema,
            analysis=analysis,
            feedback_sent=None,
            duration_ms=(time.time() - start_time) * 1000,
        )
        
        return PipelineResult(
            original_requirements=requirements,
            final_schema=result.schema,
            final_analysis=analysis,
            is_valid=analysis.is_valid if analysis else False,
            rounds=[round_data],
            total_duration_ms=round_data.duration_ms,
            tokens_used=tokens,
        )


def run_comparison_experiment(
    pipeline: SchemaRefinementPipeline,
    requirements_list: list[str],
    output_path: Optional[Path] = None,
) -> dict:
    """
    Run comparison between refinement and no-refinement approaches.
    
    Args:
        pipeline: Configured pipeline
        requirements_list: List of requirements to generate schemas for
        output_path: Optional path to save results
        
    Returns:
        Dict with comparison metrics
    """
    results = {
        "with_refinement": [],
        "without_refinement": [],
    }
    
    for i, requirements in enumerate(requirements_list):
        print(f"Processing {i+1}/{len(requirements_list)}: {requirements[:50]}...")
        
        # With refinement
        result_with = pipeline.generate(requirements)
        results["with_refinement"].append(result_with.to_dict())
        
        # Without refinement
        result_without = pipeline.generate_without_refinement(requirements)
        results["without_refinement"].append(result_without.to_dict())
        
    # Compute summary metrics
    with_valid = sum(1 for r in results["with_refinement"] if r["is_valid"])
    without_valid = sum(1 for r in results["without_refinement"] if r["is_valid"])
    total = len(requirements_list)
    
    summary = {
        "total_requirements": total,
        "with_refinement": {
            "valid_count": with_valid,
            "valid_rate": with_valid / max(total, 1),
            "avg_rounds": sum(r["num_rounds"] for r in results["with_refinement"]) / max(total, 1),
            "avg_duration_ms": sum(r["total_duration_ms"] for r in results["with_refinement"]) / max(total, 1),
        },
        "without_refinement": {
            "valid_count": without_valid,
            "valid_rate": without_valid / max(total, 1),
            "avg_duration_ms": sum(r["total_duration_ms"] for r in results["without_refinement"]) / max(total, 1),
        },
        "improvement": {
            "absolute": with_valid - without_valid,
            "relative": (with_valid - without_valid) / max(without_valid, 1),
        },
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
            
    return summary
