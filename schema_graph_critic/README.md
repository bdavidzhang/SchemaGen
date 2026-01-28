# SchemaGraph Critic 🔍

A **Neuro-Symbolic** middleware that validates LLM-generated JSON Schemas using Graph Neural Networks.

> "Standard validators only check if a schema is *technically* legal. SchemaGraph Critic checks if it's *structurally* correct."

## 🎯 The Problem

LLMs see JSON as a sequence of tokens (1D). They struggle with:
- Long-distance dependencies (closing brackets opened 400 tokens ago)
- Circular references they can't see
- Logical constraint conflicts (minItems: 10, maxItems: 3)
- Dangling $ref pointers to non-existent definitions

**Constrained decoding** solves syntax, but not **structural logic**.

## 💡 The Solution

SchemaGraph Critic treats JSON Schemas as **graphs**, not text:

```
LLM → JSON Schema → 📊 Graph → 🧠 GNN → ✓/✗ + Feedback → LLM (refinement)
```

The GNN instantly sees that Node A refers to Node Z, regardless of how far apart they are in the text.

## 🏗️ Architecture

### Heterogeneous Graph Representation

```
Nodes (Schema Elements):
├── OBJECT, ARRAY, STRING, NUMBER, INTEGER, BOOLEAN, NULL
├── REF ($ref pointers)
├── DEFINITION (in definitions/$defs)
└── LOGIC (anyOf, oneOf, allOf)

Edges (Relationships):
├── CONTAINS: Parent → Child property
├── ITEMS: Array → Items definition  
├── REFERS_TO: $ref → Target definition
├── LOGIC: anyOf/oneOf → Options
└── ADDITIONAL: additionalProperties edge
```

### SchemaGNN Model

- **Backbone**: Heterogeneous Graph Transformer (HGT) with 3-4 layers
- **Global Critic Head**: Binary classification (valid/invalid)
- **Local Debugger Head**: Per-node error probability

### Node Features (404 dimensions)

| Feature | Dimensions | Source |
|---------|------------|--------|
| Semantic embedding | 384 | MiniLM from title/description |
| Type one-hot | 11 | Node type encoding |
| Depth | 1 | Normalized nesting level |
| Constraint flags | 8 | Has min/max, pattern, required, etc. |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Parse a Schema

```python
from schema_graph_critic import SchemaGraphParser

parser = SchemaGraphParser()
graph = parser.parse("path/to/schema.json")

print(f"Nodes: {graph['schema_node'].num_nodes}")
print(f"Features: {graph['schema_node'].x.shape}")
```

### Generate Training Data

```python
from schema_graph_critic import SchemaCorruptor

corruptor = SchemaCorruptor(seed=42)

# Load your valid schemas
schemas = [...]

# Generate corrupted versions for training
dataset = corruptor.generate_dataset(
    schemas,
    valid_ratio=0.3,           # 30% valid examples
    corruptions_per_schema=3,  # 3 corrupted versions each
)
```

### Train the Model

```python
from schema_graph_critic.trainer import Trainer, TrainingConfig

config = TrainingConfig(
    hidden_dim=256,
    num_layers=3,
    num_epochs=50,
    learning_rate=1e-4,
)

trainer = Trainer(config, device="cuda")
trainer.setup()
history = trainer.train(valid_schemas)
```

### Validate & Get Feedback

```python
from schema_graph_critic import SchemaGraphParser, SchemaGNN, FeedbackTranslator

# Load trained model
model = SchemaGNN()
model.load_state_dict(torch.load("checkpoints/schema_gnn.pt"))

# Parse and analyze
parser = SchemaGraphParser()
graph = parser.parse(llm_generated_schema)

translator = FeedbackTranslator(model)
analysis = translator.analyze(graph)

if not analysis.is_valid:
    # Send feedback to LLM for refinement
    correction_prompt = translator.get_correction_prompt(
        analysis,
        original_schema_json,
        context="API response schema"
    )
    # refined_schema = llm.generate(correction_prompt)
```

### End-to-End LLM Refinement Pipeline

```python
from schema_graph_critic import SchemaRefinementPipeline, OpenAIProvider

# Configure the pipeline
llm = OpenAIProvider(model="gpt-4-turbo-preview")
pipeline = SchemaRefinementPipeline.from_checkpoint(
    "checkpoints/schema_gnn_epoch_50.pt",
    llm_provider=llm,
    max_rounds=5,
)

# Generate schema with iterative refinement
result = pipeline.generate(
    "Create a schema for an e-commerce order with items, customer info, and payment"
)

print(result.summary())
# Valid: Yes
# Rounds: 2  
# Round 0: ✗ invalid (score=42%, 234ms)
# Round 1: ✓ valid (score=94%, 189ms)
```

### Evaluate Model Performance

```python
from schema_graph_critic import Evaluator

evaluator = Evaluator.from_checkpoint("checkpoints/schema_gnn_epoch_50.pt")

# Evaluate on test schemas
result = evaluator.evaluate_from_schemas(
    test_schemas,
    valid_ratio=0.3,
    corruptions_per_schema=3,
)

print(result.summary())
# GLOBAL VALIDITY PREDICTION
#   Accuracy:  92.50%
#   Precision: 89.20%
#   Recall:    95.30%
#   F1 Score:  92.14%
```

### Compare with Baselines

```python
from schema_graph_critic import CombinedBaseline, run_baseline_evaluation

baseline = CombinedBaseline()  # Reference + Cycle + Constraint checkers
results = run_baseline_evaluation(test_examples, baseline)

print(f"Baseline F1: {results['f1']:.2%}")
```

## 📁 Project Structure

```
schema_graph_critic/
├── __init__.py          # Package exports
├── parser.py            # JSON → Heterogeneous Graph
├── model.py             # SchemaGNN (HGT + ablation variants)
├── corruptor.py         # Training data synthesis
├── trainer.py           # Training pipeline
├── translator.py        # GNN output → LLM feedback
├── evaluate.py          # Evaluation metrics (P/R/F1)
├── baselines.py         # Deterministic baselines
├── llm_pipeline.py      # End-to-end refinement loop
└── cli.py               # Command-line interface

examples/
├── schemas/             # Example JSON schemas
└── demo.py              # Interactive demonstration

tests/
├── test_parser.py       # Parser tests
├── test_corruptor.py    # Corruption tests
├── test_evaluate.py     # Evaluation tests
└── test_baselines.py    # Baseline tests
```

## 🖥️ Command-Line Interface

```bash
# Validate a schema with a trained model
schema-critic validate schema.json --model checkpoints/schema_gnn_epoch_50.pt

# Parse schema to graph representation
schema-critic parse schema.json --output graph.pt

# Generate corrupted training data
schema-critic corrupt schema.json --output corrupted/ --num 50

# Train the model
schema-critic train schemas/ --epochs 50 --hidden-dim 256

# Evaluate model on test data
schema-critic evaluate checkpoints/model.pt schemas/ --output results.json

# Run baseline comparison
schema-critic baseline schemas/ --limit 100 --output baseline_results.json
```

## 🧪 Corruption Types

The corruptor generates these error types for training:

| Type | Description | Example |
|------|-------------|---------|
| `DANGLING_REF` | $ref points to non-existent definition | `$ref: "#/definitions/Missing"` |
| `CIRCULAR_REF` | Infinite reference loop | A→B→A |
| `TYPE_MISMATCH` | Type conflicts with structure | `type: "string"` with `properties: {}` |
| `CONSTRAINT_CONFLICT` | Impossible constraints | `minItems: 10, maxItems: 3` |
| `MISSING_REQUIRED` | Required property undefined | `required: ["missing"]` |
| `INVALID_PATTERN` | Malformed regex | `pattern: "[invalid("` |
| `WRONG_ITEMS_TYPE` | Invalid items schema | `items: "not_a_schema"` |

## 📊 Demo

Run the interactive demonstration:

```bash
python examples/demo.py
```

This shows:
1. Schema → Graph parsing
2. Corruption for data synthesis  
3. Model architecture overview
4. Inference pipeline
5. Feedback generation

## 🔧 Configuration

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Model
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    
    # Loss weights
    global_loss_weight: float = 1.0  # Validity classification
    local_loss_weight: float = 0.5   # Node error detection
```

## 🔬 Ablation Studies

The package includes model variants for ablation studies:

```python
from schema_graph_critic import create_model

# Main model: Heterogeneous Graph Transformer
model_hgt = create_model("hgt", hidden_dim=256, num_layers=3)

# Ablation 1: Homogeneous GCN (treats all edges the same)
model_gcn = create_model("gcn", hidden_dim=256, num_layers=3)

# Ablation 2: Homogeneous GAT
model_gat = create_model("gat", hidden_dim=256, num_layers=3, num_heads=4)

# Ablation 3: No semantic embeddings (only structural features)
model_no_semantic = create_model("no_semantic", hidden_dim=256, num_layers=3)
```

### Model Variants

| Model | Description | Input Dim |
|-------|-------------|-----------|
| `hgt` | Heterogeneous Graph Transformer (default) | 404 |
| `gcn` | Graph Convolutional Network | 404 |
| `gat` | Graph Attention Network | 404 |
| `no_semantic` | HGT without semantic embeddings | 20 |

## 🔍 Deterministic Baselines

Compare against classical graph algorithms:

```python
from schema_graph_critic import (
    ReferenceChecker,   # Detects dangling $ref
    CycleDetector,      # Detects circular references
    ConstraintChecker,  # Detects constraint conflicts
    StandardValidator,  # jsonschema library wrapper
    CombinedBaseline,   # All of the above
)

# Individual baseline
checker = ReferenceChecker()
result = checker.validate(schema)
print(f"Valid: {result.is_valid}, Issues: {result.issues}")

# Combined baseline
baseline = CombinedBaseline()
result = baseline.validate(schema)
```

## 📈 Why This Wins

| Approach | Syntax | Logic | Feedback |
|----------|--------|-------|----------|
| `ajv` validator | ✅ | ❌ | ❌ |
| Constrained decoding | ✅ | ❌ | ❌ |
| **SchemaGraph Critic** | ✅ | ✅ | ✅ |

### Example

LLM generates a "Binary Tree" schema where `left` is defined but `right` refers to a non-existent node:

- **Constrained Decoding**: ✅ Accepts (valid JSON)
- **SchemaGraph Critic**: ❌ Rejects (REFERS_TO edge points to null)

## 📚 References

- [Heterogeneous Graph Transformer (HGT)](https://arxiv.org/abs/2003.01332)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [JSON Schema Specification](https://json-schema.org/)

## 📄 License

MIT
