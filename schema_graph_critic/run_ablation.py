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