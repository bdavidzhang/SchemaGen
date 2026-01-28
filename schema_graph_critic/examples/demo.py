#!/usr/bin/env python3
"""
Demo script for SchemaGraph Critic.

Demonstrates:
1. Parsing a JSON Schema into a heterogeneous graph
2. Corrupting schemas for training data generation
3. Running inference with the model
4. Generating feedback for LLMs
"""

import json
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_graph_critic import SchemaGraphParser, SchemaGNN, SchemaCorruptor, FeedbackTranslator


def demo_parser():
    """Demonstrate the JSON to Graph parser."""
    print("\n" + "="*60)
    print("📊 DEMO: JSON Schema to Graph Parser")
    print("="*60)
    
    # Load example schema
    schema_path = Path(__file__).parent / "schemas" / "user_profile.json"
    with open(schema_path) as f:
        schema = json.load(f)
    
    print(f"\n📄 Loaded schema: {schema.get('title', 'Unknown')}")
    print(f"   Description: {schema.get('description', 'N/A')}")
    
    # Parse to graph
    parser = SchemaGraphParser()
    graph = parser.parse(schema)
    
    print(f"\n📈 Graph Statistics:")
    print(f"   • Nodes: {graph['schema_node'].num_nodes}")
    print(f"   • Feature dimensions: {graph['schema_node'].x.shape[1]}")
    
    # Count edges by type
    edge_types = ["contains", "items", "refers_to", "logic", "additional"]
    print(f"   • Edges by type:")
    for etype in edge_types:
        edge_key = ("schema_node", etype, "schema_node")
        if edge_key in graph:
            num_edges = graph[edge_key].edge_index.shape[1]
            if num_edges > 0:
                print(f"     - {etype}: {num_edges}")
    
    # Show node paths
    print(f"\n📍 Node paths (first 10):")
    for i, path in enumerate(graph.node_paths[:10]):
        node_type = graph.node_types[i]
        print(f"   [{i}] {node_type:12} → {path}")
    if len(graph.node_paths) > 10:
        print(f"   ... and {len(graph.node_paths) - 10} more")
        
    return graph, schema


def demo_corruptor(schema: dict):
    """Demonstrate the schema corruptor."""
    print("\n" + "="*60)
    print("🔧 DEMO: Schema Corruptor (Data Synthesis)")
    print("="*60)
    
    corruptor = SchemaCorruptor(seed=42)
    
    print("\n🎯 Generating corrupted versions of the schema...")
    
    # Show each corruption type
    from schema_graph_critic.corruptor import CorruptionType
    
    for ctype in CorruptionType:
        results = corruptor.corrupt(schema.copy(), ctype, num_corruptions=1)
        if results:
            result = results[0]
            print(f"\n   {ctype.name}:")
            print(f"   └─ {result.description}")
            print(f"      Affected paths: {result.corrupted_paths}")
    
    # Generate a small dataset
    print("\n📦 Generating training dataset...")
    dataset = corruptor.generate_dataset(
        [schema],
        valid_ratio=0.3,
        corruptions_per_schema=5,
    )
    
    valid_count = sum(1 for ex in dataset if ex["is_valid"])
    invalid_count = len(dataset) - valid_count
    
    print(f"   • Total examples: {len(dataset)}")
    print(f"   • Valid schemas: {valid_count}")
    print(f"   • Corrupted schemas: {invalid_count}")
    
    return dataset


def demo_model():
    """Demonstrate the SchemaGNN model architecture."""
    print("\n" + "="*60)
    print("🧠 DEMO: SchemaGNN Model Architecture")
    print("="*60)
    
    model = SchemaGNN(
        input_dim=404,
        hidden_dim=256,
        num_heads=4,
        num_layers=3,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   • Input dimension: 404")
    print(f"     └─ Semantic embedding: 384")
    print(f"     └─ Type one-hot: 11")
    print(f"     └─ Depth: 1")
    print(f"     └─ Constraint flags: 8")
    print(f"   • Hidden dimension: 256")
    print(f"   • Number of HGT layers: 3")
    print(f"   • Attention heads: 4")
    print(f"\n📊 Parameters:")
    print(f"   • Total: {total_params:,}")
    print(f"   • Trainable: {trainable_params:,}")
    
    print(f"\n🎯 Output Heads:")
    print(f"   1. Global Critic Head → validity_score (0-1)")
    print(f"   2. Local Debugger Head → node_error_probs (per node)")
    
    return model


def demo_inference(model: SchemaGNN, graph):
    """Demonstrate model inference."""
    print("\n" + "="*60)
    print("🔮 DEMO: Model Inference (Untrained)")
    print("="*60)
    
    import torch
    model.eval()
    
    with torch.no_grad():
        output = model(graph)
    
    print(f"\n📤 Model Output:")
    print(f"   • Validity score: {output['validity_score'].item():.4f}")
    print(f"   • Node error probabilities shape: {output['node_error_probs'].shape}")
    
    # Get top error nodes
    defects = model.get_defect_nodes(graph, threshold=0.3, top_k=5)
    
    print(f"\n⚠️  Top potential defect nodes (untrained model - random):")
    for defect in defects:
        print(f"   [{defect['node_id']:2}] {defect['json_path'][:40]:40} → {defect['probability']:.2%}")
    
    return output


def demo_translator(model: SchemaGNN, graph, schema: dict):
    """Demonstrate the feedback translator."""
    print("\n" + "="*60)
    print("💬 DEMO: Feedback Translator")
    print("="*60)
    
    translator = FeedbackTranslator(
        model=model,
        validity_threshold=0.5,
        defect_threshold=0.3,
        max_defects=5,
    )
    
    analysis = translator.analyze(graph)
    
    print(f"\n📋 Analysis Summary:")
    print(f"   • Is Valid: {analysis.is_valid}")
    print(f"   • Validity Score: {analysis.validity_score:.2%}")
    print(f"   • Defects Found: {len(analysis.defects)}")
    
    print(f"\n📝 Generated Feedback Prompt:")
    print("-" * 40)
    # Truncate for demo
    feedback_lines = analysis.feedback_prompt.split("\n")[:15]
    for line in feedback_lines:
        print(f"   {line}")
    if len(analysis.feedback_prompt.split("\n")) > 15:
        print("   ...")
    print("-" * 40)
    
    # Show LLM correction prompt format
    print(f"\n🤖 LLM Correction Prompt (for integration):")
    correction_prompt = translator.get_correction_prompt(
        analysis,
        json.dumps(schema, indent=2)[:500] + "...",
        context="User profile schema for a social application"
    )
    print("-" * 40)
    for line in correction_prompt.split("\n")[:20]:
        print(f"   {line}")
    print("   ...")
    print("-" * 40)


def main():
    """Run all demos."""
    print("\n" + "🚀"*30)
    print("\n       SCHEMAGRAPH CRITIC - DEMONSTRATION")
    print("\n" + "🚀"*30)
    
    print("\nThis demo shows the core components of the SchemaGraph Critic")
    print("neuro-symbolic middleware for validating LLM-generated JSON Schemas.\n")
    
    # Run demos
    graph, schema = demo_parser()
    demo_corruptor(schema)
    model = demo_model()
    demo_inference(model, graph)
    demo_translator(model, graph, schema)
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE")
    print("="*60)
    print("""
Next Steps:
1. Collect valid JSON schemas for training
2. Run the corruptor to generate training data
3. Train the model using the Trainer class
4. Integrate with your LLM pipeline using FeedbackTranslator

See the README for detailed instructions.
""")


if __name__ == "__main__":
    main()
