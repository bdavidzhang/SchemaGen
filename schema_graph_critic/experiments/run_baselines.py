#!/usr/bin/env python3
"""
Experiment 2: Baseline Comparison

Compares deterministic baselines (reference checking, cycle detection, 
constraint validation) against the test set.
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_graph_critic import CombinedBaseline, run_baseline_evaluation
from schema_graph_critic.corruptor import SchemaCorruptor

# Configuration
SCHEMA_DIR = Path(__file__).parent.parent / "schema_miner/mined_schemas/wild_schemas"
LIMIT = 500
OUTPUT = Path(__file__).parent.parent / "results/baseline_results.json"


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
    
    # Generate test examples
    print("\nGenerating test examples...")
    corruptor = SchemaCorruptor(seed=42)
    test_examples = corruptor.generate_dataset(
        schemas,
        valid_ratio=0.3,
        corruptions_per_schema=3,
    )
    print(f"Generated {len(test_examples)} test examples")
    
    # Run individual baselines
    from schema_graph_critic.baselines import (
        ReferenceChecker,
        CycleDetector,
        ConstraintChecker,
    )
    
    all_results = {}
    
    for baseline_cls in [ReferenceChecker, CycleDetector, ConstraintChecker, CombinedBaseline]:
        baseline = baseline_cls() if baseline_cls != CombinedBaseline else baseline_cls(include_standard=False)
        print(f"\nRunning {baseline.name}...")
        
        results = run_baseline_evaluation(test_examples, baseline)
        all_results[baseline.name] = results
        
        print(f"  Accuracy:  {results['accuracy']:.2%}")
        print(f"  Precision: {results['precision']:.2%}")
        print(f"  Recall:    {results['recall']:.2%}")
        print(f"  F1 Score:  {results['f1']:.2%}")
    
    # Summary table
    print("\n" + "="*70)
    print("BASELINE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Baseline':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*70)
    for name, r in all_results.items():
        print(f"{name:<35} {r['accuracy']:.2%}        {r['precision']:.2%}        {r['recall']:.2%}        {r['f1']:.2%}")
    
    with open(OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
