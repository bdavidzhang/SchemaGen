import os
import csv
import json
import time
from datetime import datetime
from dotenv import load_dotenv

from src.config_loader import load_config
from src.llm_client import LLMClient
from src.scenarios import SCENARIOS
from src.evaluator import evaluate_submission

def run_evaluation():
    # 0. Setup
    load_dotenv() # Load environment variables
    config = load_config()
    
    # Initialize Client
    client = LLMClient(config.llm)
    
    # Create Output Directory
    os.makedirs(config.evaluation.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(config.evaluation.output_dir, f"report_{timestamp}.csv")
    
    results = []
    
    print(f"Starting Evaluation with model: {config.llm.model}")
    print(f"Total Scenarios: {len(SCENARIOS)}")
    if config.evaluation.max_scenarios:
        print(f"Limiting to first {config.evaluation.max_scenarios} scenarios.")

    # 1. Loop through scenarios
    count = 0
    for scenario in SCENARIOS:
        if config.evaluation.max_scenarios and count >= config.evaluation.max_scenarios:
            break
            
        print(f"\nEvaluating: {scenario.name} ({scenario.id})")
        
        # 2. Generate
        start_time = time.time()
        schema_str, instance_str = client.generate(scenario.prompt_template)
        latency = time.time() - start_time
        
        # 3. Evaluate
        eval_result = evaluate_submission(schema_str, instance_str, scenario)
        
        result_row = {
            "scenario_id": scenario.id,
            "scenario_name": scenario.name,
            "difficulty": scenario.difficulty,
            "track": scenario.track,
            "syntax_valid": eval_result.syntax_valid,
            "meta_schema_valid": eval_result.meta_schema_valid,
            "self_consistent": eval_result.self_consistent,
            "constraint_recall": eval_result.constraint_recall_score,
            "errors": "; ".join(eval_result.errors),
            "latency_sec": round(latency, 2)
        }
        results.append(result_row)
        
        # Log to console
        status_icon = "✅" if eval_result.self_consistent and eval_result.constraint_recall_score > 0 else "❌"
        print(f"  Result: {status_icon} | Consistent: {eval_result.self_consistent} | Recall: {eval_result.constraint_recall_score:.2f}")

        # Optional: Save individual artifacts
        if config.evaluation.save_responses:
            artifact_dir = os.path.join(config.evaluation.output_dir, f"artifacts_{timestamp}")
            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, f"{scenario.id}_schema.json"), "w") as f:
                f.write(schema_str)
            with open(os.path.join(artifact_dir, f"{scenario.id}_instance.json"), "w") as f:
                f.write(instance_str)

        count += 1

    # 4. Report
    if config.reporting.format == "csv":
        with open(report_file, "w", newline='') as csvfile:
            fieldnames = list(results[0].keys()) if results else []
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        
        print(f"\nEvaluation Complete. Report saved to: {report_file}")
    
    # Summary stats
    total = len(results)
    if total > 0:
        valid_cnt = sum(1 for r in results if r['self_consistent'])
        print(f"Summary: {valid_cnt}/{total} Scenarios Passed Self-Consistency Gate.")

if __name__ == "__main__":
    run_evaluation()
