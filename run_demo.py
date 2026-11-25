from pathlib import Path

from modelmatrix.datasets import load_dataset
from modelmatrix.evaluate import evaluate_predictions
from modelmatrix.experiments import (
    compare_runs,
    create_run_artifact,
    save_run
)

# 1. LOAD THE VERSIONED BENCHMARK
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "benchmarks" / "arithmetic_v1.json"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"

dataset = load_dataset(DATASET_PATH)


# 2. RUN A SIMULATED EVALUATION
baseline_predictions = ["4", "9", "7", "12", "0", "5", "undefined", "144", "1", "60"]
candidate_predictions = ["4", "8", "7", "12", "0", "5", "0", "144", "1", "120"]


# 3. EVALUATE AND SAVE EACH RUN
baseline_report = evaluate_predictions(dataset.examples, baseline_predictions)
candidate_report = evaluate_predictions(dataset.examples, candidate_predictions)

baseline_run = create_run_artifact(dataset, baseline_report)
candidate_run = create_run_artifact(dataset, candidate_report)

baseline_path = save_run(baseline_run, RUNS_DIR)
candidate_path = save_run(candidate_run, RUNS_DIR)


# 4. COMPARE THE EXPERIMENTS
comparison = compare_runs(baseline_run, candidate_run)


# 5. DISPLAY RESULTS AND ARTIFACT LOCATIONS
print(f"Dataset: {dataset.name} v{dataset.version}")
print(f"Dataset SHA-256: {dataset.sha256}")
print()

print("Baseline")
print(f"  Run ID: {baseline_run.run_id}")
print(f"  Accuracy: {baseline_run.accuracy:.1%}")

print("Candidate")
print(f"  Run ID: {candidate_run.run_id}")
print(f"  Accuracy: {candidate_run.accuracy:.1%}")
print()

print("Comparison")
print(
    f"  Accuracy change: "
    f"{comparison.accuracy_delta * 100:+.1f} percentage points"
)
print(f"  Improvements: {comparison.improvements}")
print(f"  Regressions: {comparison.regressions}")
print()

print(f"Saved baseline: {baseline_path.relative_to(PROJECT_ROOT)}")
print(f"Saved candidate: {candidate_path.relative_to(PROJECT_ROOT)}")

for result in candidate_run.results:
    status = "PASS" if result.correct else "FAIL"

    print(
        f"[{status}] [{result.id}] {result.input} | "
        f"Expected: {result.expected} | "
        f"Predicted: {result.predicted}"
    )
