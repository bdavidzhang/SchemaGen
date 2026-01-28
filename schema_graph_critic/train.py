#!/usr/bin/env python3
"""
Training script for SchemaGNN using mined wild schemas.

Usage:
    python train.py                          # Train with default settings
    python train.py --epochs 100 --device cuda
    python train.py --schema-dir path/to/schemas
"""

import argparse
import json
from pathlib import Path

import torch

from schema_graph_critic.trainer import Trainer, TrainingConfig


def load_schemas_from_directory(schema_dir: Path, limit: int = None) -> list[dict]:
    """
    Load JSON schema files from a directory.
    
    Args:
        schema_dir: Directory containing .json schema files
        limit: Maximum number of schemas to load (None for all)
        
    Returns:
        List of parsed schema dictionaries
    """
    schemas = []
    schema_files = list(schema_dir.glob("*.json"))
    
    if limit:
        schema_files = schema_files[:limit]
    
    print(f"Loading {len(schema_files)} schema files...")
    
    for schema_path in schema_files:
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
                
            # Basic validation - must be a dict
            if isinstance(schema, dict):
                schemas.append(schema)
                
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # Skip invalid files
            continue
        except Exception as e:
            print(f"Warning: Could not load {schema_path.name}: {e}")
            continue
    
    print(f"Successfully loaded {len(schemas)} valid schemas")
    return schemas


def main():
    parser = argparse.ArgumentParser(description="Train SchemaGNN model")
    
    # Data arguments
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path("schema_miner/mined_schemas/wild_schemas"),
        help="Directory containing JSON schema files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of schemas to use (for quick testing)"
    )
    
    # Model arguments
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    
    # Data generation arguments
    parser.add_argument("--valid-ratio", type=float, default=0.3)
    parser.add_argument("--corruptions-per-schema", type=int, default=3)
    
    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=5)
    
    args = parser.parse_args()
    
    # Load schemas
    print(f"\n{'='*60}")
    print("SchemaGNN Training")
    print(f"{'='*60}\n")
    
    if not args.schema_dir.exists():
        print(f"Error: Schema directory not found: {args.schema_dir}")
        return
        
    schemas = load_schemas_from_directory(args.schema_dir, args.limit)
    
    if len(schemas) < 10:
        print(f"Error: Not enough schemas ({len(schemas)}). Need at least 10.")
        return
    
    # Create config
    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        corruptions_per_schema=args.corruptions_per_schema,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
    )
    
    # Print configuration
    print("Configuration:")
    print(f"  Schemas:               {len(schemas)}")
    print(f"  Training examples:     ~{len(schemas) * (args.corruptions_per_schema + int(args.valid_ratio))}")
    print(f"  Device:                {args.device}")
    print(f"  Hidden dim:            {config.hidden_dim}")
    print(f"  Num layers:            {config.num_layers}")
    print(f"  Num heads:             {config.num_heads}")
    print(f"  Epochs:                {config.num_epochs}")
    print(f"  Learning rate:         {config.learning_rate}")
    print(f"  Valid ratio:           {config.valid_ratio}")
    print(f"  Corruptions/schema:    {config.corruptions_per_schema}")
    print(f"  Checkpoint dir:        {config.checkpoint_dir}")
    print()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Create trainer and train
    trainer = Trainer(config, device=args.device)
    trainer.setup()
    
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print()
    
    # Train!
    print("Starting training...\n")
    history = trainer.train(schemas, val_split=args.val_split)
    
    # Print final results
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    
    final = history[-1]
    print(f"Final train loss:      {final.train_loss:.4f}")
    if final.val_accuracy is not None:
        print(f"Final val accuracy:    {final.val_accuracy:.2%}")
        print(f"Final val node acc:    {final.val_node_accuracy:.2%}")
    
    print(f"\nCheckpoints saved to: {config.checkpoint_dir}/")


if __name__ == "__main__":
    main()
