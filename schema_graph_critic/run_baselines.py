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