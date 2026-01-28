#!/usr/bin/env python3
"""
Experiment 5: Per-Corruption-Type Analysis

Detailed breakdown of model performance on each corruption type:
- DANGLING_REF
- CIRCULAR_REF
- TYPE_MISMATCH
- CONSTRAINT_CONFLICT
- MISSING_REQUIRED
- INVALID_PATTERN
- WRONG_ITEMS_TYPE
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from schema_graph_critic import Evaluator, SchemaGNN, SchemaGraphParser

# Configuration
CHECKPOINT = Path(__file__).parent.parent / "checkpoints/schema_gnn_epoch_50.pt"
SCHEMA_DIR = Path(__file__).parent.parent / "schema_miner/mined_schemas/wild_schemas"
LIMIT = 500
OUTPUT = Path(__file__).parent.parent / "results/per_corruption_results.json"


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    
    # Load schemas
    print(f"Loading schemas from {SCHEMA_DIR}...")
    schemas = []
    for path in list(SCHEMA_DIR.glob("**/*.json"))[:LIMIT]:
        try:
            with open(path) as f:
                schemas.append(json.load(f))
        except Exception:
            continue
    
    print(f"Loaded {len(schemas)} schemas")
    
    if not schemas:
        print("ERROR: No schemas found!")
        sys.exit(1)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print(f"Loading model from {CHECKPOINT}...")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SchemaGNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluate
    print("\nRunning evaluation...")
    parser = SchemaGraphParser(device=device)
    evaluator = Evaluator(model, parser, device=device)
    result = evaluator.evaluate_from_schemas(schemas, seed=42)
    
    # Print per-corruption breakdown
    print("\n" + "="*80)
    print("PER-CORRUPTION-TYPE PERFORMANCE")
    print("="*80)
    print(f"{'Corruption Type':<25} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*80)
    
    for corruption_type, metrics in sorted(result.per_corruption_metrics.items()):
        print(f"{corruption_type:<25} {metrics.true_positives:<6} {metrics.false_positives:<6} {metrics.false_negatives:<6} {metrics.precision:.2%}        {metrics.recall:.2%}        {metrics.f1:.2%}")
    
    # Summary
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"Global Accuracy:  {result.global_metrics.accuracy:.2%}")
    print(f"Global Precision: {result.global_metrics.precision:.2%}")
    print(f"Global Recall:    {result.global_metrics.recall:.2%}")
    print(f"Global F1:        {result.global_metrics.f1:.2%}")
    print(f"\nNode Accuracy:    {result.node_metrics.accuracy:.2%}")
    print(f"Node F1:          {result.node_metrics.f1:.2%}")
    
    with open(OUTPUT, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
