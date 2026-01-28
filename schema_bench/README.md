# SchemaGen Bench

SchemaGen Bench evaluates an LLM's ability to **design data contracts** (JSON Schemas) and **fulfill them** (JSON Instances). 

A sophisticated agent must not only generate valid JSON but define the structure correctly to avoid ambiguity and data errors.

## Core Philosophy

Existing benchmarks test compliance ("Follow this schema").  
**SchemaGen Bench** tests design ("Create a schema for X, and generate data for it").

It evaluates:
1.  **The Architect**: Can the model create a valid, specific, and correct JSON Schema?
2.  **The Builder**: Can the model generate a strictly valid JSON instance that adheres to its own schema?

## Architecture

### Tracks (Scenarios)
We organize scenarios by complexity:
*   **Track A**: Structural Complexity (Flat, Nested, Recursive, Polymorphic)
*   **Track B**: Constraint Hardness (Strict types, regex formats, numerical bounds)
*   **Track C**: Ambiguity Resolution (Does the model infer reasonable defaults?)

### Evaluation Gates
The evaluation pipeline filters output through "Gates":
1.  **Gate 1: Syntax & Meta-Validity**: Is it valid JSON? Is the Schema a valid JSON Schema (Draft 2020-12)?
2.  **Gate 2: Self-Consistency**: Does the generated JSON Instance validate against the generated Schema? (Includes strict format checking).
3.  **Gate 3: Semantic Alignment**: Does the schema actually fulfill the requirements? (Constraint Recall, Specificity).

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure:
   - Copy `.env.example` to `.env` and set your API Key.
   - Edit `config/config.yaml` to change model or parameters.

3. Run the evaluation:
   ```bash
   python main.py
   ```

## Project Structure

*   `config/`: Configuration files.
*   `src/llm_client.py`: Interface for OpenAI (and others).
*   `src/evaluator.py`: Core validation logic.
*   `src/scenarios.py`: The benchmark dataset.
*   `main.py`: The main execution entry point.

## Usage

To evaluate a real LLM:

1.  Import `SCENARIOS` from `src.scenarios`.
2.  Iterate through them and feed `scenario.prompt_template` to your model.
3.  Parse the model output to separate the JSON Schema and the JSON Instance.
4.  Pass them to `evaluate_submission`.
