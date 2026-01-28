# Experiment Procedure

This document outlines the steps to run all experiments for the SchemaGraph Critic paper.

## Prerequisites

### 1. Activate the Virtual Environment

```bash
cd /Users/zhangbocheng/code/projects/schema_gen/schema_graph_critic
source .venv/bin/activate
```

### 2. Install Missing Dependencies (if needed)

```bash
uv pip install torch-geometric
uv pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "from schema_graph_critic import SchemaGNN, Evaluator, CombinedBaseline; print('All imports OK!')"
```

---

## Experiment 1: Model Evaluation (Table 8)

Evaluate the trained SchemaGNN model on test data to get precision, recall, and F1 scores.

### Using CLI

```bash
# Evaluate on wild schemas (limit to 500 for reasonable runtime)
python -m schema_graph_critic.cli evaluate \
    checkpoints/schema_gnn_epoch_50.pt \
    schema_miner/mined_schemas/wild_schemas \
    --limit 500 \
    --valid-ratio 0.3 \
    --corruptions 3 \
    --output results/evaluation_results.json

# View results
cat results/evaluation_results.json
```

### Using Python Script

```python
# Run this as: python run_evaluation.py
import json
from pathlib import Path
from schema_graph_critic import Evaluator, SchemaGraphParser, SchemaGNN
import torch

# Configuration
CHECKPOINT = "checkpoints/schema_gnn_epoch_50.pt"
SCHEMA_DIR = Path("schema_miner/mined_schemas/wild_schemas")
LIMIT = 500
OUTPUT = "results/evaluation_results.json"

# Create output directory
Path("results").mkdir(exist_ok=True)

# Load schemas
schemas = []
for path in list(SCHEMA_DIR.glob("**/*.json"))[:LIMIT]:
    try:
        with open(path) as f:
            schemas.append(json.load(f))
    except:
        continue

print(f"Loaded {len(schemas)} schemas")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(CHECKPOINT, map_location=device)
model = SchemaGNN()
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluate
parser = SchemaGraphParser(device=device)
evaluator = Evaluator(model, parser, device=device)
result = evaluator.evaluate_from_schemas(
    schemas,
    valid_ratio=0.3,
    corruptions_per_schema=3,
    seed=42,
)

# Print and save
print(result.summary())
with open(OUTPUT, "w") as f:
    json.dump(result.to_dict(), f, indent=2)

print(f"\nResults saved to {OUTPUT}")
```

---

## Experiment 2: Baseline Comparison (Table 8)

Compare SchemaGNN against deterministic baselines (reference checking, cycle detection, constraint validation).

### Using CLI

```bash
python -m schema_graph_critic.cli baseline \
    schema_miner/mined_schemas/wild_schemas \
    --limit 500 \
    --valid-ratio 0.3 \
    --corruptions 3 \
    --output results/baseline_results.json
```

### Using Python Script

```python
# Run this as: python run_baselines.py
import json
from pathlib import Path
from schema_graph_critic import CombinedBaseline, run_baseline_evaluation
from schema_graph_critic.corruptor import SchemaCorruptor

SCHEMA_DIR = Path("schema_miner/mined_schemas/wild_schemas")
LIMIT = 500
OUTPUT = "results/baseline_results.json"

Path("results").mkdir(exist_ok=True)

# Load schemas
schemas = []
for path in list(SCHEMA_DIR.glob("**/*.json"))[:LIMIT]:
    try:
        with open(path) as f:
            schemas.append(json.load(f))
    except:
        continue

print(f"Loaded {len(schemas)} schemas")

# Generate test examples
corruptor = SchemaCorruptor(seed=42)
test_examples = corruptor.generate_dataset(
    schemas,
    valid_ratio=0.3,
    corruptions_per_schema=3,
)
print(f"Generated {len(test_examples)} test examples")

# Run baseline
baseline = CombinedBaseline(include_standard=True)
results = run_baseline_evaluation(test_examples, baseline)

print("\n" + "="*50)
print("BASELINE RESULTS")
print("="*50)
print(f"Accuracy:  {results['accuracy']:.2%}")
print(f"Precision: {results['precision']:.2%}")
print(f"Recall:    {results['recall']:.2%}")
print(f"F1 Score:  {results['f1']:.2%}")

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT}")
```

---

## Experiment 3: Ablation Studies (Table comparing model variants)

Compare HGT (main model) vs GCN, GAT, and no-semantic variants.

### Python Script

```python
# Run this as: python run_ablation.py
import json
from pathlib import Path
import torch
from schema_graph_critic import create_model, SchemaGraphParser
from schema_graph_critic.trainer import Trainer, TrainingConfig
from schema_graph_critic.evaluate import Evaluator

SCHEMA_DIR = Path("schema_miner/mined_schemas/wild_schemas")
LIMIT = 300  # Use fewer for faster ablation runs
OUTPUT = "results/ablation_results.json"

Path("results").mkdir(exist_ok=True)

# Load schemas
schemas = []
for path in list(SCHEMA_DIR.glob("**/*.json"))[:LIMIT]:
    try:
        with open(path) as f:
            schemas.append(json.load(f))
    except:
        continue

print(f"Loaded {len(schemas)} schemas")

# Split into train/test
split = int(len(schemas) * 0.8)
train_schemas = schemas[:split]
test_schemas = schemas[split:]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

results = {}

# Train and evaluate each model variant
for model_type in ["hgt", "gcn", "gat"]:
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} model...")
    print("="*50)
    
    config = TrainingConfig(
        hidden_dim=128,  # Smaller for faster ablation
        num_layers=2,
        num_epochs=20,
        checkpoint_dir=f"checkpoints/ablation_{model_type}",
    )
    
    trainer = Trainer(config, device=device)
    trainer.setup()
    
    # Replace model with ablation variant
    if model_type != "hgt":
        trainer.model = create_model(model_type, hidden_dim=128, num_layers=2).to(device)
    
    history = trainer.train(train_schemas, val_split=0.1)
    
    # Evaluate
    parser = SchemaGraphParser(device=device)
    evaluator = Evaluator(trainer.model, parser, device=device)
    eval_result = evaluator.evaluate_from_schemas(test_schemas, seed=42)
    
    results[model_type] = {
        "global_f1": eval_result.global_metrics.f1,
        "global_accuracy": eval_result.global_metrics.accuracy,
        "node_f1": eval_result.node_metrics.f1,
        "final_train_loss": history[-1].train_loss,
    }
    
    print(f"\n{model_type.upper()} Results:")
    print(f"  Global F1: {eval_result.global_metrics.f1:.2%}")
    print(f"  Node F1:   {eval_result.node_metrics.f1:.2%}")

# Summary table
print("\n" + "="*60)
print("ABLATION STUDY RESULTS")
print("="*60)
print(f"{'Model':<15} {'Global F1':<12} {'Node F1':<12} {'Accuracy':<12}")
print("-"*60)
for model_type, r in results.items():
    print(f"{model_type.upper():<15} {r['global_f1']:.2%}        {r['node_f1']:.2%}        {r['global_accuracy']:.2%}")

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT}")
```

---

## Experiment 4: LLM Refinement Pipeline (End-to-End Evaluation)

Test the full generation → validation → refinement loop with an LLM.

### Prerequisites

Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"
```

### Python Script

```python
# Run this as: python run_llm_pipeline.py
import json
from pathlib import Path
from schema_graph_critic import (
    SchemaRefinementPipeline,
    OpenAIProvider,
    # AnthropicProvider,  # Alternative
    run_comparison_experiment,
)

CHECKPOINT = "checkpoints/schema_gnn_epoch_50.pt"
OUTPUT = "results/llm_pipeline_results.json"

Path("results").mkdir(exist_ok=True)

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

# Configure pipeline
llm = OpenAIProvider(model="gpt-4-turbo-preview")
# llm = AnthropicProvider(model="claude-3-5-sonnet-20241022")  # Alternative

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
```

---

## Experiment 5: Per-Corruption-Type Analysis

Detailed breakdown of model performance on each corruption type.

### Python Script

```python
# Run this as: python run_per_corruption.py
import json
from pathlib import Path
from schema_graph_critic import Evaluator, SchemaGNN, SchemaGraphParser
import torch

CHECKPOINT = "checkpoints/schema_gnn_epoch_50.pt"
SCHEMA_DIR = Path("schema_miner/mined_schemas/wild_schemas")
LIMIT = 500
OUTPUT = "results/per_corruption_results.json"

Path("results").mkdir(exist_ok=True)

# Load schemas
schemas = []
for path in list(SCHEMA_DIR.glob("**/*.json"))[:LIMIT]:
    try:
        with open(path) as f:
            schemas.append(json.load(f))
    except:
        continue

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(CHECKPOINT, map_location=device)
model = SchemaGNN()
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluate
parser = SchemaGraphParser(device=device)
evaluator = Evaluator(model, parser, device=device)
result = evaluator.evaluate_from_schemas(schemas, seed=42)

# Print per-corruption breakdown
print("="*70)
print("PER-CORRUPTION-TYPE PERFORMANCE")
print("="*70)
print(f"{'Corruption Type':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-"*70)

for corruption_type, metrics in sorted(result.per_corruption_metrics.items()):
    print(f"{corruption_type:<25} {metrics.precision:.2%}        {metrics.recall:.2%}        {metrics.f1:.2%}")

with open(OUTPUT, "w") as f:
    json.dump(result.to_dict(), f, indent=2)

print(f"\nResults saved to {OUTPUT}")
```

---

## Quick Reference: All Commands

```bash
# Activate environment
source .venv/bin/activate

# Create results directory
mkdir -p results

# Run all experiments
python run_evaluation.py      # Experiment 1: Model Evaluation
python run_baselines.py       # Experiment 2: Baseline Comparison
python run_ablation.py        # Experiment 3: Ablation Studies
python run_llm_pipeline.py    # Experiment 4: LLM Pipeline (requires API key)
python run_per_corruption.py  # Experiment 5: Per-Corruption Analysis

# Or use CLI
python -m schema_graph_critic.cli evaluate checkpoints/schema_gnn_epoch_50.pt schema_miner/mined_schemas/wild_schemas --limit 500 --output results/eval.json
python -m schema_graph_critic.cli baseline schema_miner/mined_schemas/wild_schemas --limit 500 --output results/baseline.json
```

---

## Expected Output Format

Results will be saved as JSON files in the `results/` directory:

- `evaluation_results.json` - Global and node-level metrics
- `baseline_results.json` - Deterministic baseline performance
- `ablation_results.json` - HGT vs GCN vs GAT comparison
- `llm_pipeline_results.json` - With vs without refinement
- `per_corruption_results.json` - Per-corruption-type breakdown

These can be directly used to populate Tables 7 and 8 in the paper.
