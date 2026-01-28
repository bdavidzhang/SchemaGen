#!/bin/bash
# Run all experiments for SchemaGraph Critic paper

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "SchemaGraph Critic Experiments"
echo "=========================================="
echo ""

cd "$PROJECT_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Create results directory
mkdir -p results

echo ""
echo "=========================================="
echo "Experiment 1: Model Evaluation"
echo "=========================================="
python experiments/run_evaluation.py

echo ""
echo "=========================================="
echo "Experiment 2: Baseline Comparison"
echo "=========================================="
python experiments/run_baselines.py

echo ""
echo "=========================================="
echo "Experiment 3: Per-Corruption Analysis"
echo "=========================================="
python experiments/run_per_corruption.py

echo ""
echo "=========================================="
echo "Experiment 4: Ablation Studies"
echo "=========================================="
echo "NOTE: This trains new models and takes longer"
read -p "Run ablation studies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python experiments/run_ablation.py
else
    echo "Skipping ablation studies"
fi

echo ""
echo "=========================================="
echo "Experiment 5: LLM Pipeline"
echo "=========================================="
if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ]; then
    python experiments/run_llm_pipeline.py
else
    echo "Skipping LLM pipeline (no API key set)"
    echo "Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this experiment"
fi

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results saved in: $PROJECT_DIR/results/"
ls -la results/
