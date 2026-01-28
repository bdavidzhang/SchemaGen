# SchemaGen Bench — Current State Analysis & Improvement Roadmap

> **Analysis Date:** January 11, 2026  
> **Version Analyzed:** Initial Implementation (v0.1)

---

## Executive Summary

SchemaGen Bench is a novel LLM benchmark that evaluates a model's ability to **design data contracts** (JSON Schema) and **fulfill them** (JSON Instance). Unlike compliance benchmarks that test "can you follow this schema," SchemaGen tests "can you design a schema and then follow it"—a fundamentally harder task that reveals architectural reasoning capabilities.

The current implementation is a **solid proof-of-concept** with a clean architecture, but it has significant room for expansion in scenario coverage, evaluation depth, and provider support.

---

## 1. Current Architecture

### 1.1 Component Overview

| Component | File | Purpose |
|-----------|------|---------|
| Entry Point | `main.py` | Orchestrates evaluation loop |
| Scenarios | `src/scenarios.py` | Benchmark dataset (6 scenarios) |
| Evaluator | `src/evaluator.py` | 4-gate validation pipeline |
| LLM Client | `src/llm_client.py` | OpenAI API wrapper with mock fallback |
| Config | `src/config_loader.py` | YAML-based configuration |
| Models | `src/models.py` | Pydantic data structures |

### 1.2 Evaluation Pipeline (Gates)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────────────┐
│  Gate 1         │     │  Gate 2          │     │  Gate 3         │     │  Gate 4            │
│  JSON Syntax    │────▶│  Meta-Schema     │────▶│ Self-Consistency│────▶│  Semantic Analysis │
│  Parse both     │     │  Is it a valid   │     │ Instance vs     │     │  Constraint Recall │
│  artifacts      │     │  JSON Schema?    │     │ Schema match?   │     │  (keyword-based)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └────────────────────┘
```

### 1.3 Scenario Coverage

Currently **6 scenarios** across 3 tracks:

| ID | Name | Track | Difficulty |
|----|------|-------|------------|
| `scen_001` | Simple App Configuration | Structural: Flat | Easy |
| `scen_002` | User Profile with Constraints | Structural: Nested | Medium |
| `scen_003` | Recursive File System | Structural: Recursive | Recursive |
| `scen_004` | Polymorphic UI Components | Structural: Polymorphic | Hard |
| `scen_005` | Strict Product SKU | Constraint Hardness | Medium |
| `scen_006` | Ambiguous User Event | Ambiguity Resolution | Easy |

---

## 2. Strengths

### ✅ Clean, Modular Architecture
- Pydantic models enforce type safety
- Clear separation between LLM client, evaluator, and scenarios
- YAML configuration makes experimentation easy

### ✅ Novel Benchmark Concept
- Tests schema *design* ability, not just compliance
- Self-consistency check is a unique differentiator
- Captures "architect + builder" duality

### ✅ Format-Aware Validation
- Uses `jsonschema` with format checkers enabled
- Validates against Draft 2020-12 meta-schema
- Mock mode allows testing without API keys

### ✅ Good Developer Experience
- CSV reports for easy analysis
- Artifact saving for debugging
- Clear console output with status icons

---

## 3. Weaknesses & Gaps

### 🔴 Critical Issues

#### 3.1 Naive Constraint Recall Metric
The current implementation uses **string matching** to check if constraints exist:

```python
hits = sum(1 for c in scenario.required_constraints if c in schema_str)
```

**Problems:**
- `"minimum"` in the string doesn't mean it's applied to the *correct* field
- Doesn't verify constraint *values* (e.g., minimum should be 1024, not just present)
- Can be gamed by including keywords in `$comment` or `description`

#### 3.2 No Gold Schema Comparison
There's no reference schema to compare against. The benchmark only checks:
1. Is it syntactically valid?
2. Does the instance match the schema?

But **not**:
- Does the schema correctly interpret the requirements?
- Are the constraints correct (right fields, right values)?

#### 3.3 Specificity Score Not Implemented
The `EvaluationResult` model defines `specificity_score` but it's never calculated:

```python
specificity_score: float = 0.0  # Unused
```

This metric was designed to measure how well-constrained the schema is.

#### 3.4 Limited Error Recovery
If JSON parsing fails, evaluation stops entirely. The benchmark could benefit from:
- Attempting to extract JSON from markdown blocks
- Partial credit for syntactically valid schema even if instance fails

### 🟡 Moderate Issues

#### 3.5 Only 6 Scenarios
The scenario set is too small to reliably differentiate models. Recommended minimum: **20-30 scenarios** with multiple variations per track.

#### 3.6 Single Provider Support
Only OpenAI is supported. Missing:
- Anthropic Claude
- Google Gemini
- Local models (Ollama, vLLM)
- Azure OpenAI

#### 3.7 No Retry Logic for Transient Failures
`max_retries` is defined in config but never used:

```yaml
max_retries: 3  # Unused
```

#### 3.8 No Token/Cost Tracking
No visibility into:
- Input/output token counts
- Cost per scenario
- Context window utilization

#### 3.9 Missing Gold Keys Validation
`gold_keys` is defined in scenarios but never used in evaluation:

```python
gold_keys=["port", "host", "debug_mode"]  # Unused
```

This should validate that the schema and instance include expected fields.

### 🟢 Minor Issues

#### 3.10 No Parallel Execution
Scenarios run sequentially. For large scenario sets, parallel execution would significantly reduce evaluation time.

#### 3.11 CSV-Only Reporting
JSON output option exists in config but isn't implemented:

```yaml
format: "csv"  # json option not implemented
```

#### 3.12 No Version Pinning
`requirements.txt` lacks version constraints, risking dependency drift.

---

## 4. Improvement Roadmap

### Phase 1: Core Metric Improvements (Priority: High)

#### 4.1 Implement Structural Constraint Validation
Replace string matching with AST traversal of the schema:

```python
def check_constraint_at_path(schema: dict, path: list[str], constraint: str, expected_value=None):
    """Verify constraint exists at specific path with optional value check."""
    node = schema
    for key in path:
        node = node.get("properties", {}).get(key, {})
    if constraint not in node:
        return False
    if expected_value is not None:
        return node[constraint] == expected_value
    return True
```

**Scenario Enhancement:**
```python
required_constraints=[
    {"path": ["port"], "constraint": "minimum", "value": 1024},
    {"path": ["port"], "constraint": "maximum", "value": 65535},
]
```

#### 4.2 Add Gold Keys Validation
Implement the unused `gold_keys` check:

```python
def validate_gold_keys(schema: dict, instance: dict, gold_keys: list[str]) -> float:
    """Returns ratio of expected keys present in both schema and instance."""
    schema_props = set(schema.get("properties", {}).keys())
    instance_keys = set(instance.keys())
    
    # Recursively check nested keys (e.g., "address.street")
    present = sum(1 for k in gold_keys if k in schema_props and k in instance_keys)
    return present / len(gold_keys) if gold_keys else 1.0
```

#### 4.3 Implement Specificity Score
Measure how well-constrained the schema is:

```python
def calculate_specificity(schema: dict) -> float:
    """Ratio of constrained properties to total properties."""
    CONSTRAINT_KEYWORDS = {"minimum", "maximum", "pattern", "format", "enum", "const", "minLength", "maxLength"}
    
    total_props = 0
    constrained_props = 0
    
    def walk(node):
        nonlocal total_props, constrained_props
        if "properties" in node:
            for prop_name, prop_schema in node["properties"].items():
                total_props += 1
                if any(k in prop_schema for k in CONSTRAINT_KEYWORDS):
                    constrained_props += 1
                walk(prop_schema)
    
    walk(schema)
    return constrained_props / total_props if total_props > 0 else 0.0
```

---

### Phase 2: Scenario Expansion (Priority: High)

#### 4.4 Add More Scenarios Per Track
Target: **30+ scenarios** with this distribution:

| Track | Current | Target | Examples to Add |
|-------|---------|--------|-----------------|
| Structural: Flat | 1 | 5 | ENV vars, CLI flags, feature toggles |
| Structural: Nested | 1 | 5 | API responses, GraphQL types, ORM entities |
| Structural: Recursive | 1 | 4 | AST nodes, org charts, comment threads |
| Structural: Polymorphic | 1 | 4 | Union types, discriminated unions, API error types |
| Constraint Hardness | 1 | 6 | Credit cards, phone numbers, URLs, semantic versions |
| Ambiguity Resolution | 1 | 6 | Underspecified prompts, contradictory requirements |

#### 4.5 Add Adversarial Scenarios
Test edge cases:
- Conflicting constraints (e.g., "must be > 100 and < 50")
- Extremely long schemas (100+ properties)
- Unicode and special characters
- Schema references to external definitions

---

### Phase 3: Multi-Provider Support (Priority: Medium)

#### 4.6 Abstract LLM Interface

```python
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> tuple[str, str]:
        pass

class OpenAIClient(BaseLLMClient):
    ...

class AnthropicClient(BaseLLMClient):
    ...

class OllamaClient(BaseLLMClient):
    ...
```

#### 4.7 Add Provider Configurations

```yaml
llm:
  provider: "anthropic"
  model: "claude-3-opus-20240229"
  api_key_env_var: "ANTHROPIC_API_KEY"
```

---

### Phase 4: Robustness & Observability (Priority: Medium)

#### 4.8 Implement Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_with_retry(self, prompt: str) -> tuple[str, str]:
    ...
```

#### 4.9 Add Token & Cost Tracking

```python
result_row = {
    ...
    "input_tokens": response.usage.prompt_tokens,
    "output_tokens": response.usage.completion_tokens,
    "cost_usd": calculate_cost(response.usage, self.config.model),
}
```

#### 4.10 JSON Extraction from Markdown
Handle models that wrap output in ```json blocks:

```python
import re

def extract_json(content: str) -> str:
    # Try to extract from markdown code block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    if match:
        return match.group(1).strip()
    return content
```

---

### Phase 5: Developer Experience (Priority: Low)

#### 4.11 Add JSON Reporting

```python
if config.reporting.format == "json":
    with open(report_file.replace(".csv", ".json"), "w") as f:
        json.dump(results, f, indent=2)
```

#### 4.12 Parallel Scenario Execution

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(evaluate_scenario, s): s for s in SCENARIOS}
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
```

#### 4.13 Pin Dependencies

```text
pydantic>=2.5.0,<3.0.0
jsonschema>=4.20.0,<5.0.0
pyyaml>=6.0.1,<7.0.0
openai>=1.6.0,<2.0.0
python-dotenv>=1.0.0,<2.0.0
tenacity>=8.2.0,<9.0.0
```

#### 4.14 Add CLI with Click/Typer

```python
import typer

app = typer.Typer()

@app.command()
def run(
    model: str = typer.Option("gpt-4-turbo", help="Model to evaluate"),
    max_scenarios: int = typer.Option(None, help="Limit scenarios"),
    provider: str = typer.Option("openai", help="LLM provider"),
):
    ...
```

---

## 5. Sample Results Analysis

Based on the existing run (`report_20260109_140614.csv`):

| Metric | Value |
|--------|-------|
| Total Scenarios | 6 |
| Self-Consistency Pass Rate | 100% (6/6) |
| Constraint Recall | 100% (all scenarios) |
| Average Latency | 14.7s |

**Observation:** GPT-4-turbo achieved perfect scores on all current scenarios. This suggests:
1. The scenarios may be too easy for state-of-the-art models
2. The constraint recall metric is too lenient (string matching)
3. More adversarial/complex scenarios are needed to differentiate models

---

## 6. Recommended Next Steps

### Immediate (This Week)
1. [ ] Implement structural constraint validation (4.1)
2. [ ] Add gold keys validation (4.2)
3. [ ] Pin dependency versions (4.13)

### Short-Term (2-4 Weeks)
4. [ ] Expand scenarios to 20+ (4.4)
5. [ ] Add Anthropic Claude support (4.6, 4.7)
6. [ ] Implement retry logic (4.8)
7. [ ] Add JSON extraction from markdown (4.10)

### Medium-Term (1-2 Months)
8. [ ] Implement specificity score (4.3)
9. [ ] Add adversarial scenarios (4.5)
10. [ ] Add token/cost tracking (4.9)
11. [ ] Add parallel execution (4.12)

### Long-Term
12. [ ] CLI interface (4.14)
13. [ ] Web dashboard for results visualization
14. [ ] Integration with evaluation harnesses (lm-eval, HELM)

---

## 7. Appendix: Scenario Ideas for Expansion

### Structural: Flat
- `cli_args`: Command-line argument parser
- `env_vars`: Environment variable configuration
- `feature_flags`: Boolean feature toggles with metadata

### Structural: Nested
- `graphql_type`: GraphQL schema with nested resolvers
- `orm_entity`: Database ORM model with relationships
- `api_response`: REST API response with metadata wrapper

### Structural: Recursive
- `ast_node`: Abstract syntax tree with expressions
- `org_chart`: Hierarchical organization with manager references
- `thread_comment`: Nested comment thread

### Constraint Hardness
- `credit_card`: Luhn-validated card numbers
- `phone_intl`: International phone number with country codes
- `semver`: Semantic versioning with pre-release tags
- `ip_cidr`: IP address with CIDR notation

### Ambiguity Resolution
- `underspec_user`: "Create a user object" (no fields specified)
- `contradictory`: "Age must be positive and less than 0"
- `implicit_types`: "Store a date" (string or integer epoch?)

---

*Document generated for SchemaGen Bench development planning.*
