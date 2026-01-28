#!/usr/bin/env python3
"""
Extended training script for SchemaGNN using both wild and repository schemas.

This script aggregates schemas from multiple sources:
- Wild schemas (~2000)
- SchemaStore (~775)
- Kubernetes JSON Schema (1 version, ~1000)
- Azure Resource Manager Schemas (~3000)
- JSON Schema Test Suite (~300)
- Metadata Schema (~300)

Usage:
    python train_extended.py                    # Train with all sources
    python train_extended.py --sources wild schemastore kubernetes
    python train_extended.py --max-per-source 500  # Limit per source
    python train_extended.py --device cuda --epochs 100
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch

from schema_graph_critic.trainer import Trainer, TrainingConfig


@dataclass
class SchemaSource:
    """Configuration for a schema source."""
    name: str
    path: Path
    pattern: str = "*.json"
    recursive: bool = True
    max_schemas: Optional[int] = None
    

# Define available schema sources
SCHEMA_SOURCES = {
    "wild": SchemaSource(
        name="Wild Schemas",
        path=Path("schema_miner/mined_schemas/wild_schemas"),
        pattern="*.json",
        recursive=False,
    ),
    "schemastore": SchemaSource(
        name="SchemaStore",
        path=Path("schema_miner/mined_schemas/repositories/schemastore/src/schemas/json"),
        pattern="*.json",
        recursive=False,
    ),
    "kubernetes": SchemaSource(
        name="Kubernetes JSON Schema",
        # Use only master-standalone to avoid version duplicates
        path=Path("schema_miner/mined_schemas/repositories/kubernetes-json-schema/master-standalone"),
        pattern="*.json",
        recursive=False,
        # Skip _definitions.json and all.json which are meta-files
    ),
    "azure": SchemaSource(
        name="Azure Resource Manager Schemas",
        path=Path("schema_miner/mined_schemas/repositories/azure-resource-manager-schemas/schemas"),
        pattern="*.json",
        recursive=True,
    ),
    "json_schema_test": SchemaSource(
        name="JSON Schema Test Suite",
        path=Path("schema_miner/mined_schemas/repositories/JSON-Schema-Test-Suite"),
        pattern="*.json",
        recursive=True,
    ),
    "metadata": SchemaSource(
        name="Metadata Schema",
        path=Path("schema_miner/mined_schemas/repositories/metadata-schema"),
        pattern="*.json",
        recursive=True,
    ),
}


def is_valid_json_schema(schema: dict) -> bool:
    """
    Check if a dictionary looks like a valid JSON Schema.
    
    Filters out:
    - Non-dict values
    - Empty schemas
    - Package.json, tsconfig, etc. (not schemas)
    - Test data (not schemas)
    """
    if not isinstance(schema, dict):
        return False
    
    if not schema:
        return False
    
    # Must have some schema-like properties
    schema_indicators = {
        "$schema", "type", "properties", "items", "allOf", "anyOf", 
        "oneOf", "$ref", "definitions", "$defs", "required", "enum",
        "const", "pattern", "format", "minimum", "maximum", "minLength",
        "maxLength", "minItems", "maxItems", "additionalProperties",
    }
    
    has_schema_property = any(key in schema for key in schema_indicators)
    
    if not has_schema_property:
        return False
    
    # Filter out common non-schema files
    if schema.get("name") and schema.get("version") and schema.get("dependencies"):
        # Looks like package.json
        return False
    
    if schema.get("compilerOptions"):
        # Looks like tsconfig.json
        return False
        
    return True


def load_schemas_from_source(
    source: SchemaSource,
    max_schemas: Optional[int] = None,
    seen_hashes: Optional[set] = None,
    verbose: bool = True,
) -> tuple[list[dict], set[str]]:
    """
    Load schemas from a source, deduplicating by content hash.
    
    Args:
        source: Schema source configuration
        max_schemas: Maximum schemas to load from this source
        seen_hashes: Set of already-seen content hashes (for cross-source dedup)
        verbose: Print progress
        
    Returns:
        Tuple of (schemas list, updated seen_hashes set)
    """
    if seen_hashes is None:
        seen_hashes = set()
        
    schemas = []
    
    if not source.path.exists():
        if verbose:
            print(f"  Warning: Path not found: {source.path}")
        return schemas, seen_hashes
    
    # Collect schema files
    if source.recursive:
        schema_files = list(source.path.rglob(source.pattern))
    else:
        schema_files = list(source.path.glob(source.pattern))
    
    # Shuffle for variety if limiting
    if max_schemas and len(schema_files) > max_schemas:
        random.shuffle(schema_files)
    
    loaded = 0
    skipped_invalid = 0
    skipped_duplicate = 0
    skipped_error = 0
    
    for schema_path in schema_files:
        if max_schemas and loaded >= max_schemas:
            break
            
        # Skip meta-files
        if schema_path.name in {"_definitions.json", "all.json", "package.json", 
                                "package-lock.json", "tsconfig.json", "jsconfig.json"}:
            continue
            
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                content = f.read()
                schema = json.loads(content)
            
            # Validate it's a schema
            if not is_valid_json_schema(schema):
                skipped_invalid += 1
                continue
                
            # Deduplicate by content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                skipped_duplicate += 1
                continue
                
            seen_hashes.add(content_hash)
            schemas.append(schema)
            loaded += 1
            
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped_error += 1
            continue
        except Exception as e:
            skipped_error += 1
            continue
    
    if verbose:
        print(f"  {source.name}: loaded {loaded}, "
              f"skipped {skipped_invalid} invalid, "
              f"{skipped_duplicate} duplicates, "
              f"{skipped_error} errors")
    
    return schemas, seen_hashes


def load_all_schemas(
    sources: list[str],
    max_per_source: Optional[int] = None,
    max_total: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load schemas from multiple sources with deduplication.
    
    Args:
        sources: List of source names to use
        max_per_source: Max schemas per source
        max_total: Max total schemas
        seed: Random seed for reproducibility
        
    Returns:
        List of unique schemas
    """
    random.seed(seed)
    
    all_schemas = []
    seen_hashes = set()
    
    print(f"\nLoading schemas from {len(sources)} sources...")
    
    for source_name in sources:
        if source_name not in SCHEMA_SOURCES:
            print(f"  Warning: Unknown source '{source_name}', skipping")
            continue
            
        source = SCHEMA_SOURCES[source_name]
        schemas, seen_hashes = load_schemas_from_source(
            source,
            max_schemas=max_per_source,
            seen_hashes=seen_hashes,
        )
        all_schemas.extend(schemas)
    
    # Optionally limit total
    if max_total and len(all_schemas) > max_total:
        random.shuffle(all_schemas)
        all_schemas = all_schemas[:max_total]
    
    print(f"\nTotal unique schemas: {len(all_schemas)}")
    return all_schemas


def main():
    parser = argparse.ArgumentParser(
        description="Train SchemaGNN with extended schema sources"
    )
    
    # Source selection
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["wild", "schemastore", "kubernetes"],
        choices=list(SCHEMA_SOURCES.keys()),
        help="Schema sources to use (default: wild schemastore kubernetes)"
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Maximum schemas per source (for balancing)"
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Maximum total schemas"
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
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("SchemaGNN Extended Training")
    print(f"{'='*60}")
    
    # Load schemas from all sources
    schemas = load_all_schemas(
        sources=args.sources,
        max_per_source=args.max_per_source,
        max_total=args.max_total,
        seed=args.seed,
    )
    
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
    
    # Estimate training examples
    est_examples = len(schemas) * (args.corruptions_per_schema + int(args.valid_ratio))
    
    print(f"\nConfiguration:")
    print(f"  Sources:               {', '.join(args.sources)}")
    print(f"  Total schemas:         {len(schemas)}")
    print(f"  Estimated examples:    ~{est_examples}")
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
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"
    
    # Create trainer and train
    trainer = Trainer(config, device=args.device)
    trainer.setup()
    
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print()
    
    # Train
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
