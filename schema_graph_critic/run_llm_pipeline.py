# Run this as: python run_llm_pipeline.py
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from schema_graph_critic import (
    SchemaRefinementPipeline,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    run_comparison_experiment,
)

# Load API keys from .env file
load_dotenv()

CHECKPOINT = "checkpoints/schema_gnn_epoch_50.pt"
OUTPUT = "results/llm_pipeline_results.json"

Path("results").mkdir(exist_ok=True)

# Check for API keys
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not openai_key and not anthropic_key and not gemini_key:
    print("ERROR: No API key found!")
    print("Create a .env file with one of the following:")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY") 
    print("  - GOOGLE_API_KEY (or GEMINI_API_KEY)")
    print("\nExample .env file:")
    print('  OPENAI_API_KEY=sk-...')
    print('  ANTHROPIC_API_KEY=sk-ant-...')
    print('  GOOGLE_API_KEY=AI...')
    exit(1)

# Test requirements
requirements = [
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

# Configure pipeline with available API key (priority: OpenAI > Anthropic > Gemini)
# if openai_key:
#     print("Using OpenAI API (gpt-4-turbo-preview)")
#     llm = OpenAIProvider(model="gpt-4-turbo-preview", api_key=openai_key)
if anthropic_key:
    print("Using Anthropic API claude-sonnet-4-5)")
    llm = AnthropicProvider(model="claude-sonnet-4-5", api_key=anthropic_key)
# else:
#     print("Using Google Gemini API (gemini-1.5-pro)")
#     llm = GeminiProvider(model="gemini-1.5-pro", api_key=gemini_key)

pipeline = SchemaRefinementPipeline.from_checkpoint(
    CHECKPOINT,
    llm_provider=llm,
    max_rounds=5,
    device="cpu",
)

# Run comparison experiment
print("Running LLM pipeline comparison...")
print("This compares: with refinement vs. without refinement\n")

summary = run_comparison_experiment(
    pipeline,
    requirements,
    output_path=Path(OUTPUT),
)

print("\n" + "="*60)
print("LLM PIPELINE RESULTS")
print("="*60)
print(f"\nWith Refinement:")
print(f"  Valid schemas: {summary['with_refinement']['valid_count']}/{summary['total_requirements']}")
print(f"  Valid rate:    {summary['with_refinement']['valid_rate']:.1%}")
print(f"  Avg rounds:    {summary['with_refinement']['avg_rounds']:.1f}")

print(f"\nWithout Refinement:")
print(f"  Valid schemas: {summary['without_refinement']['valid_count']}/{summary['total_requirements']}")
print(f"  Valid rate:    {summary['without_refinement']['valid_rate']:.1%}")

print(f"\nImprovement:")
print(f"  Absolute: +{summary['improvement']['absolute']} valid schemas")
print(f"  Relative: +{summary['improvement']['relative']:.1%}")

print(f"\nFull results saved to {OUTPUT}")