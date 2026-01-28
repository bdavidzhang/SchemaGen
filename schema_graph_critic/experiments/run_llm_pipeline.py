#!/usr/bin/env python3
"""
Experiment 4: LLM Refinement Pipeline

Tests the full generation → validation → refinement loop.
Compares with and without SchemaGraph Critic feedback.

Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY in .env file or environment
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from schema_graph_critic import (
    SchemaRefinementPipeline,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    run_comparison_experiment,
)

# Configuration
CHECKPOINT = Path(__file__).parent.parent / "checkpoints/schema_gnn_epoch_50.pt"
OUTPUT = Path(__file__).parent.parent / "results/llm_pipeline_results.json"

# Test requirements
REQUIREMENTS = [
    "Create a schema for a user profile with name, email, age (0-150), and optional address",
    "Create a schema for an e-commerce order with items, quantities, prices, and customer info",
    "Create a schema for a blog post with title, content, author, tags, and comments",
    "Create a schema for a REST API error response with error code, message, and details",
    "Create a schema for a configuration file with database settings and feature flags",
    "Create a schema for a calendar event with title, start/end times, attendees, and recurrence",
    "Create a schema for a product catalog with categories, products, variants, and pricing",
    "Create a schema for a social media post with author, content, reactions, and replies",
    "Create a schema for a payment transaction with amount, currency, status, and metadata",
    "Create a schema for a resume/CV with education, experience, skills, and projects",
]


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    
    # Check for API keys (from .env or environment)
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not openai_key and not anthropic_key and not gemini_key:
        print("ERROR: No API key found!")
        print("Create a .env file in the project root with one of:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("  GOOGLE_API_KEY=AI...")
        sys.exit(1)
    
    # Configure LLM provider (priority: OpenAI > Anthropic > Gemini)
    if openai_key:
        print("Using OpenAI API (gpt-4-turbo-preview)")
        llm = OpenAIProvider(model="gpt-4-turbo-preview", api_key=openai_key)
    elif anthropic_key:
        print("Using Anthropic API (claude-3-5-sonnet)")
        llm = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key=anthropic_key)
    else:
        print("Using Google Gemini API (gemini-1.5-pro)")
        llm = GeminiProvider(model="gemini-1.5-pro", api_key=gemini_key)
    
    # Create pipeline
    print(f"Loading model from {CHECKPOINT}...")
    pipeline = SchemaRefinementPipeline.from_checkpoint(
        CHECKPOINT,
        llm_provider=llm,
        max_rounds=5,
        device="cpu",
    )
    
    # Run comparison
    print(f"\nRunning comparison experiment on {len(REQUIREMENTS)} requirements...")
    print("This compares: with refinement vs. without refinement\n")
    
    summary = run_comparison_experiment(
        pipeline,
        REQUIREMENTS,
        output_path=OUTPUT,
    )
    
    # Print results
    print("\n" + "="*60)
    print("LLM PIPELINE RESULTS")
    print("="*60)
    
    print(f"\nWith Refinement (using SchemaGraph Critic):")
    print(f"  Valid schemas: {summary['with_refinement']['valid_count']}/{summary['total_requirements']}")
    print(f"  Valid rate:    {summary['with_refinement']['valid_rate']:.1%}")
    print(f"  Avg rounds:    {summary['with_refinement']['avg_rounds']:.1f}")
    print(f"  Avg time:      {summary['with_refinement']['avg_duration_ms']:.0f}ms")
    
    print(f"\nWithout Refinement (single-shot):")
    print(f"  Valid schemas: {summary['without_refinement']['valid_count']}/{summary['total_requirements']}")
    print(f"  Valid rate:    {summary['without_refinement']['valid_rate']:.1%}")
    print(f"  Avg time:      {summary['without_refinement']['avg_duration_ms']:.0f}ms")
    
    print(f"\nImprovement from refinement:")
    print(f"  Absolute: +{summary['improvement']['absolute']} valid schemas")
    print(f"  Relative: +{summary['improvement']['relative']:.1%}")
    
    print(f"\nFull results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
