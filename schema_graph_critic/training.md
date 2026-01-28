# SchemaGNN Training Procedure

This document describes how to train the SchemaGNN model using the extended dataset that includes both wild schemas and repository schemas.

## Dataset Overview

| Source | Location | Count | Description |
|--------|----------|-------|-------------|
| Wild Schemas | `schema_miner/mined_schemas/wild_schemas/` | ~2,000 | Schemas mined from GitHub |
| SchemaStore | `repositories/schemastore/src/schemas/json/` | ~775 | High-quality, curated schemas |
| Kubernetes | `repositories/kubernetes-json-schema/master-standalone/` | ~1,100 | K8s resource schemas |
| Azure ARM | `repositories/azure-resource-manager-schemas/schemas/` | ~3,200 | Azure Resource Manager |
| JSON Schema Test Suite | `repositories/JSON-Schema-Test-Suite/` | ~300 | Official test schemas |
| Metadata Schema | `repositories/metadata-schema/` | ~300 | Metadata standards |

**Total potential schemas: ~7,700** (after deduplication)

## Quick Start

### 1. Setup Environment

```bash
cd schema_graph_critic
source .venv/bin/activate
```

### 2. Train with Default Sources (Recommended)

Uses wild schemas + SchemaStore + Kubernetes (~4,000 schemas):

```bash
python train_extended.py --device mps --epochs 100
```

### 3. Train with All Sources

Uses all available sources (~7,700 schemas):

```bash
python train_extended.py \
    --sources wild schemastore kubernetes azure json_schema_test metadata \
    --device mps \
    --epochs 100
```

## Training Options

### Source Selection

Choose which schema sources to include:

```bash
# Minimal (fast, for testing)
python train_extended.py --sources wild --epochs 10

# Balanced (recommended for experiments)
python train_extended.py --sources wild schemastore kubernetes

# Full dataset
python train_extended.py --sources wild schemastore kubernetes azure json_schema_test metadata
```

### Limiting Dataset Size

For faster iteration or memory constraints:

```bash
# Limit each source to 500 schemas (balanced training)
python train_extended.py --max-per-source 500

# Limit total dataset
python train_extended.py --max-total 3000

# Both limits
python train_extended.py --max-per-source 1000 --max-total 5000
```

### Model Architecture

```bash
# Larger model (more capacity)
python train_extended.py \
    --hidden-dim 512 \
    --num-layers 4 \
    --num-heads 8

# Smaller model (faster training)
python train_extended.py \
    --hidden-dim 128 \
    --num-layers 2 \
    --num-heads 4
```

### Training Hyperparameters

```bash
python train_extended.py \
    --epochs 100 \
    --lr 1e-4 \
    --batch-size 32 \
    --val-split 0.15 \
    --valid-ratio 0.3 \
    --corruptions-per-schema 3
```

### Device Selection

```bash
# Apple Silicon (M1/M2/M3)
python train_extended.py --device mps

# NVIDIA GPU
python train_extended.py --device cuda

# CPU only
python train_extended.py --device cpu
```

## Training Configurations by Goal

### Configuration 1: Quick Validation (10 min)

Test that everything works:

```bash
python train_extended.py \
    --sources wild \
    --max-per-source 200 \
    --epochs 5 \
    --device mps
```

### Configuration 2: Standard Training (1-2 hours)

Good balance of data and training time:

```bash
python train_extended.py \
    --sources wild schemastore kubernetes \
    --epochs 50 \
    --device mps
```

### Configuration 3: Full Training (4-8 hours)

Maximum data, thorough training:

```bash
python train_extended.py \
    --sources wild schemastore kubernetes azure json_schema_test metadata \
    --epochs 100 \
    --hidden-dim 512 \
    --num-layers 4 \
    --device mps
```

### Configuration 4: Balanced Sources (Recommended)

Balance dataset to prevent domain bias:

```bash
python train_extended.py \
    --sources wild schemastore kubernetes azure \
    --max-per-source 1000 \
    --epochs 75 \
    --device mps
```

## Understanding the Training Process

### Data Generation Pipeline

1. **Load schemas** from selected sources
2. **Deduplicate** by content hash (avoids duplicates across sources)
3. **Validate** each file is a proper JSON Schema
4. **Generate training examples**:
   - `valid_ratio` (30% default) of examples are uncorrupted valid schemas
   - Remaining examples have synthetic corruptions applied
5. **Split** into train/validation sets

### Corruption Types Applied

The `SchemaCorruptor` applies these corruptions for negative examples:

- **Dangling references**: `$ref` pointing to non-existent definitions
- **Circular references**: Infinite reference loops
- **Type conflicts**: Invalid type combinations
- **Constraint conflicts**: e.g., `minimum > maximum`
- **Invalid patterns**: Malformed regex patterns
- **Missing required**: Required fields not in properties

### Training Objectives

The model optimizes two loss functions:

1. **Global validity loss**: Binary classification - is the schema valid?
2. **Local node loss**: Per-node error detection - which nodes are corrupted?

## Evaluating After Training

After training completes, evaluate on the test set:

```bash
# Run evaluation
python -m schema_graph_critic.cli evaluate \
    --checkpoint checkpoints/schema_gnn_epoch_100.pt \
    --test-dir schema_miner/mined_schemas/wild_schemas \
    --output results/extended_evaluation.json

# Run baselines for comparison
python -m schema_graph_critic.cli baseline \
    --test-dir schema_miner/mined_schemas/wild_schemas \
    --output results/extended_baselines.json
```

## Tips for Better Results

### 1. Balance Your Sources

If one source dominates (e.g., Azure has 3000+ schemas), use `--max-per-source`:

```bash
python train_extended.py --max-per-source 1000
```

### 2. Increase Corruption Diversity

More corruption types = better error detection:

```bash
python train_extended.py --corruptions-per-schema 5
```

### 3. Use Early Stopping (Manual)

Watch validation accuracy - if it plateaus, you can stop early:

```
Epoch 50: val_acc=92.5%
Epoch 55: val_acc=92.6%  
Epoch 60: val_acc=92.4%  <- Plateau, consider stopping
```

### 4. Learning Rate Schedule

The training uses cosine annealing. For longer training, start with higher LR:

```bash
python train_extended.py --epochs 200 --lr 3e-4
```

## Troubleshooting

### Out of Memory

- Reduce `--hidden-dim` (try 128)
- Reduce `--max-total` to limit dataset
- Use CPU if GPU memory is insufficient

### Slow Training

- Use `--device mps` or `--device cuda`
- Reduce dataset with `--max-per-source`
- Reduce `--epochs` for faster iteration

### Low Validation Accuracy

- Increase dataset diversity (add more sources)
- Increase model capacity (`--hidden-dim 512`)
- Check if data is too homogeneous (e.g., all Kubernetes schemas)

### Overfitting (train acc >> val acc)

- Add more data sources
- Increase `--dropout` (try 0.2)
- Reduce `--hidden-dim`
- Use `--val-split 0.2` for more validation data

## Output Files

Training produces:

```
checkpoints/
├── schema_gnn_epoch_5.pt
├── schema_gnn_epoch_10.pt
├── ...
└── schema_gnn_epoch_100.pt  # Final model
```

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `metrics`: Training metrics at that epoch
- `config`: Training configuration
