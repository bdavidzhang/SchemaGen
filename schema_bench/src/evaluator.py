import json
import jsonschema
from jsonschema import validate
from jsonschema.validators import validator_for
from .models import BenchmarkScenario, EvaluationResult

def evaluate_submission(schema_str: str, instance_str: str, scenario: BenchmarkScenario) -> EvaluationResult:
    result = EvaluationResult(scenario_id=scenario.id)
    result.errors = []

    # 1. Syntax Check
    schema = None
    instance = None
    try:
        schema = json.loads(schema_str)
        instance = json.loads(instance_str)
        result.syntax_valid = True
    except json.JSONDecodeError as e:
        result.errors.append(f"JSON Syntax Error: {str(e)}")
        return result

    # 2. Meta-Schema Check (Is the schema a valid schema?)
    try:
        # Check against Draft 2020-12 or latest supported by jsonschema
        cls = validator_for(schema)
        cls.check_schema(schema)
        result.meta_schema_valid = True
    except jsonschema.exceptions.SchemaError as e:
        result.errors.append(f"Invalid JSON Schema: {str(e)}")
        return result
    except Exception as e:
        result.errors.append(f"Meta-schema validation error: {str(e)}")
        return result

    # 3. Self-Consistency (Does instance match schema?)
    try:
        # We need to enable format checking to catch things like invalid emails or dates
        validator_cls = validator_for(schema)
        # Check if the validator class has a FORMAT_CHECKER (most standard ones do)
        format_checker = validator_cls.FORMAT_CHECKER if hasattr(validator_cls, "FORMAT_CHECKER") else None
        
        validator = validator_cls(schema, format_checker=format_checker)
        validator.validate(instance)
        result.self_consistent = True
    except jsonschema.exceptions.ValidationError as e:
        result.errors.append(f"Instance does not match Schema: {str(e)}")
        # We don't return here, we can still evaluate semantic alignment if possible, 
        # basically if the schema is valid we can check it.
    
    # 4. Semantic Analysis
    # Metric 1: Constraint Recall
    # Simple keyword string check (can be improved with AST parsing of the schema later)
    if scenario.required_constraints:
        # This is a naive check. A robust check would traverse the schema dict.
        # But for the blueprint, string matching key constraints or schema keywords works for now.
        # However, checking if "minimum" is *in the string* doesn't mean it's applied to the right field.
        # For now, let's stick to the prompt's suggestion but acknowledge limitations.
        hits = sum(1 for c in scenario.required_constraints if c in schema_str)
        result.constraint_recall_score = hits / len(scenario.required_constraints)
    else:
        result.constraint_recall_score = 1.0

    # Metric 2: Specificity Score (simplified implementation from prompt)
    # (Count of constrained fields) / (Total fields) - requires traversing code.
    # For now, let's count occurrences of specific keywords vs generic ones.
    # We will skip complex traversal for this initial version.
    
    return result
