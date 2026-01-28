#!/usr/bin/env python3
"""
Command-line interface for SchemaGraph Critic.

Usage:
    schema-critic validate schema.json
    schema-critic parse schema.json --output graph.pt
    schema-critic corrupt schema.json --output corrupted/
    schema-critic train schemas/ --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def cmd_validate(args):
    """Validate a JSON schema and show analysis."""
    from .parser import SchemaGraphParser
    from .model import SchemaGNN
    from .translator import FeedbackTranslator
    
    # Load schema
    with open(args.schema) as f:
        schema = json.load(f)
        
    print(f"📄 Validating: {args.schema}")
    
    # Parse to graph
    parser = SchemaGraphParser()
    graph = parser.parse(schema)
    
    print(f"   Nodes: {graph['schema_node'].num_nodes}")
    
    # Load model if provided
    if args.model:
        model = SchemaGNN()
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        translator = FeedbackTranslator(model)
        analysis = translator.analyze(graph)
        
        print(f"\n{'✅' if analysis.is_valid else '❌'} {analysis.summary}")
        
        if not analysis.is_valid:
            print(f"\n{analysis.feedback_prompt}")
    else:
        print("\n⚠️  No model provided (--model). Only structural parsing performed.")
        print("   Run 'schema-critic train' first to create a model.")
        

def cmd_parse(args):
    """Parse a schema to graph and optionally save."""
    from .parser import SchemaGraphParser
    
    with open(args.schema) as f:
        schema = json.load(f)
        
    parser = SchemaGraphParser()
    graph = parser.parse(schema)
    
    print(f"📊 Graph created from {args.schema}")
    print(f"   Nodes: {graph['schema_node'].num_nodes}")
    print(f"   Features shape: {graph['schema_node'].x.shape}")
    
    if args.output:
        torch.save(graph, args.output)
        print(f"   Saved to: {args.output}")
        

def cmd_corrupt(args):
    """Generate corrupted versions of schemas."""
    from .corruptor import SchemaCorruptor
    
    schema_path = Path(args.schema)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(schema_path) as f:
        schema = json.load(f)
        
    corruptor = SchemaCorruptor(seed=args.seed)
    
    print(f"🔧 Corrupting: {schema_path}")
    
    dataset = corruptor.generate_dataset(
        [schema],
        valid_ratio=0.2,
        corruptions_per_schema=args.num,
    )
    
    # Save examples
    for i, example in enumerate(dataset):
        output_path = output_dir / f"example_{i:04d}.json"
        with open(output_path, "w") as f:
            json.dump({
                "schema": example["schema"],
                "is_valid": example["is_valid"],
                "corrupted_paths": example["corrupted_paths"],
                "corruption_type": example["corruption_type"],
                "description": example["description"],
            }, f, indent=2)
            
    print(f"   Generated {len(dataset)} examples")
    print(f"   Valid: {sum(1 for e in dataset if e['is_valid'])}")
    print(f"   Corrupted: {sum(1 for e in dataset if not e['is_valid'])}")
    print(f"   Output: {output_dir}/")


def cmd_train(args):
    """Train the SchemaGNN model."""
    from .trainer import Trainer, TrainingConfig
    
    schema_dir = Path(args.schemas)
    
    # Load all schemas
    schemas = []
    for schema_path in schema_dir.glob("**/*.json"):
        try:
            with open(schema_path) as f:
                schemas.append(json.load(f))
        except Exception as e:
            print(f"⚠️  Skipping {schema_path}: {e}")
            
    if not schemas:
        print("❌ No valid JSON schemas found!")
        sys.exit(1)
        
    print(f"📚 Loaded {len(schemas)} schemas from {schema_dir}")
    
    config = TrainingConfig(
        num_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        checkpoint_dir=args.output,
    )
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"🖥️  Device: {device}")
    
    trainer = Trainer(config, device=device)
    trainer.setup()
    
    print(f"\n🚀 Starting training...")
    history = trainer.train(schemas, val_split=0.1)
    
    print(f"\n✅ Training complete!")
    print(f"   Final train loss: {history[-1].train_loss:.4f}")
    if history[-1].val_accuracy:
        print(f"   Final val accuracy: {history[-1].val_accuracy:.2%}")
    print(f"   Checkpoints: {args.output}/")


def cmd_evaluate(args):
    """Evaluate a trained model on test data."""
    from .evaluate import Evaluator, EvaluationResult
    from .parser import SchemaGraphParser
    from .model import SchemaGNN
    
    checkpoint_path = Path(args.model)
    schema_dir = Path(args.schemas)
    
    # Load schemas
    schemas = []
    for schema_path in schema_dir.glob("**/*.json"):
        try:
            with open(schema_path) as f:
                schemas.append(json.load(f))
        except Exception:
            continue
            
    if args.limit:
        schemas = schemas[:args.limit]
        
    print(f"📚 Loaded {len(schemas)} schemas from {schema_dir}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"🖥️  Device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SchemaGNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    
    parser = SchemaGraphParser(device=device)
    evaluator = Evaluator(model, parser, device=device)
    
    print(f"\n🔍 Evaluating...")
    result = evaluator.evaluate_from_schemas(
        schemas,
        valid_ratio=args.valid_ratio,
        corruptions_per_schema=args.corruptions,
        seed=args.seed,
    )
    
    print(result.summary())
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n📄 Results saved to {output_path}")


def cmd_baseline(args):
    """Run baseline comparison."""
    from .baselines import CombinedBaseline, run_baseline_evaluation
    from .corruptor import SchemaCorruptor
    
    schema_dir = Path(args.schemas)
    
    # Load schemas
    schemas = []
    for schema_path in schema_dir.glob("**/*.json"):
        try:
            with open(schema_path) as f:
                schemas.append(json.load(f))
        except Exception:
            continue
            
    if args.limit:
        schemas = schemas[:args.limit]
        
    print(f"📚 Loaded {len(schemas)} schemas from {schema_dir}")
    
    # Generate test examples
    corruptor = SchemaCorruptor(seed=args.seed)
    test_examples = corruptor.generate_dataset(
        schemas,
        valid_ratio=args.valid_ratio,
        corruptions_per_schema=args.corruptions,
    )
    
    print(f"🔧 Generated {len(test_examples)} test examples")
    
    # Run baselines
    baseline = CombinedBaseline(include_standard=not args.skip_standard)
    
    print(f"\n🔍 Running baseline evaluation...")
    results = run_baseline_evaluation(test_examples, baseline)
    
    print(f"\n{'='*50}")
    print("BASELINE RESULTS")
    print(f"{'='*50}")
    print(f"Baseline: {results['baseline']}")
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall:    {results['recall']:.2%}")
    print(f"F1 Score:  {results['f1']:.2%}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SchemaGraph Critic - Validate LLM-generated JSON Schemas with GNNs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a JSON schema")
    validate_parser.add_argument("schema", help="Path to JSON schema file")
    validate_parser.add_argument("--model", "-m", help="Path to trained model checkpoint")
    validate_parser.set_defaults(func=cmd_validate)
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse schema to graph")
    parse_parser.add_argument("schema", help="Path to JSON schema file")
    parse_parser.add_argument("--output", "-o", help="Output path for graph (.pt)")
    parse_parser.set_defaults(func=cmd_parse)
    
    # Corrupt command
    corrupt_parser = subparsers.add_parser("corrupt", help="Generate corrupted schemas")
    corrupt_parser.add_argument("schema", help="Path to valid JSON schema")
    corrupt_parser.add_argument("--output", "-o", default="corrupted", help="Output directory")
    corrupt_parser.add_argument("--num", "-n", type=int, default=10, help="Number of corruptions")
    corrupt_parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    corrupt_parser.set_defaults(func=cmd_corrupt)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the SchemaGNN model")
    train_parser.add_argument("schemas", help="Directory containing JSON schemas")
    train_parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    train_parser.add_argument("--num-layers", type=int, default=3, help="Number of HGT layers")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--output", "-o", default="checkpoints", help="Checkpoint directory")
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    train_parser.set_defaults(func=cmd_train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on test data")
    eval_parser.add_argument("model", help="Path to model checkpoint")
    eval_parser.add_argument("schemas", help="Directory containing JSON schemas")
    eval_parser.add_argument("--limit", "-l", type=int, help="Limit number of schemas")
    eval_parser.add_argument("--valid-ratio", type=float, default=0.3, help="Ratio of valid examples")
    eval_parser.add_argument("--corruptions", "-c", type=int, default=3, help="Corruptions per schema")
    eval_parser.add_argument("--seed", "-s", type=int, default=123, help="Random seed")
    eval_parser.add_argument("--output", "-o", help="Output JSON file for results")
    eval_parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Run baseline comparison")
    baseline_parser.add_argument("schemas", help="Directory containing JSON schemas")
    baseline_parser.add_argument("--limit", "-l", type=int, help="Limit number of schemas")
    baseline_parser.add_argument("--valid-ratio", type=float, default=0.3, help="Ratio of valid examples")
    baseline_parser.add_argument("--corruptions", "-c", type=int, default=3, help="Corruptions per schema")
    baseline_parser.add_argument("--seed", "-s", type=int, default=123, help="Random seed")
    baseline_parser.add_argument("--skip-standard", action="store_true", help="Skip jsonschema validation")
    baseline_parser.add_argument("--output", "-o", help="Output JSON file for results")
    baseline_parser.set_defaults(func=cmd_baseline)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
