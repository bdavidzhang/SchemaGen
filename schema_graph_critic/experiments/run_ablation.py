#!/usr/bin/env python3
"""
Experiment 3: Ablation Studies

Compares model variants:
- HGT (Heterogeneous Graph Transformer) - main model
- GCN (Graph Convolutional Network) - homogeneous baseline
- GAT (Graph Attention Network) - homogeneous baseline
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from schema_graph_critic import create_model, SchemaGraphParser
from schema_graph_critic.trainer import Trainer, TrainingConfig
from schema_graph_critic.evaluate import Evaluator

# Configuration
SCHEMA_DIR = Path(__file__).parent.parent / "schema_miner/mined_schemas/wild_schemas"
LIMIT = 300  # Fewer for faster ablation
OUTPUT = Path(__file__).parent.parent / "results/ablation_results.json"
CHECKPOINT_BASE = Path(__file__).parent.parent / "checkpoints"


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
    
    # Split into train/test
    split = int(len(schemas) * 0.8)
    train_schemas = schemas[:split]
    test_schemas = schemas[split:]
    print(f"Train: {len(train_schemas)}, Test: {len(test_schemas)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    results = {}
    
    # Train and evaluate each model variant
    for model_type in ["hgt", "gcn", "gat"]:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model...")
        print("="*60)
        
        config = TrainingConfig(
            hidden_dim=128,
            num_layers=2,
            num_epochs=20,
            checkpoint_dir=str(CHECKPOINT_BASE / f"ablation_{model_type}"),
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
            "global_precision": eval_result.global_metrics.precision,
            "global_recall": eval_result.global_metrics.recall,
            "node_f1": eval_result.node_metrics.f1,
            "node_accuracy": eval_result.node_metrics.accuracy,
            "final_train_loss": history[-1].train_loss,
            "final_val_accuracy": history[-1].val_accuracy,
        }
        
        print(f"\n{model_type.upper()} Results:")
        print(f"  Global F1:  {eval_result.global_metrics.f1:.2%}")
        print(f"  Node F1:    {eval_result.node_metrics.f1:.2%}")
        print(f"  Accuracy:   {eval_result.global_metrics.accuracy:.2%}")
    
    # Summary table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"{'Model':<10} {'Global F1':<12} {'Global Acc':<12} {'Node F1':<12} {'Node Acc':<12}")
    print("-"*70)
    for model_type, r in results.items():
        print(f"{model_type.upper():<10} {r['global_f1']:.2%}        {r['global_accuracy']:.2%}        {r['node_f1']:.2%}        {r['node_accuracy']:.2%}")
    
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
