#!/usr/bin/env python3
"""
Experiment 1: Model Evaluation

Evaluates the trained SchemaGNN model on test data.
Produces precision, recall, F1 for global and node-level predictions.
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from schema_graph_critic import Evaluator, SchemaGraphParser, SchemaGNN

# Configuration
CHECKPOINT = Path(__file__).parent.parent / "checkpoints/schema_gnn_epoch_50.pt"
SCHEMA_DIR = Path(__file__).parent.parent / "schema_miner/mined_schemas/wild_schemas"
LIMIT = 500
OUTPUT = Path(__file__).parent.parent / "results/evaluation_results.json"


def main():
    # Create output directory
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
    result = evaluator.evaluate_from_schemas(
        schemas,
        valid_ratio=0.3,
        corruptions_per_schema=3,
        seed=42,
    )
    
    # Print and save
    print("\n" + result.summary())
    
    with open(OUTPUT, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
