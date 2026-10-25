from pathlib import Path

from modelmatrix.datasets import load_dataset
from modelmatrix.evaluate import evaluate_predictions

# 1. LOAD THE VERSIONED BENCHMARK
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "benchmarks" / "arithmetic_v1.json"

dataset = load_dataset(DATASET_PATH)


# 2. RUN A SIMULATED EVALUATION
predictions = ["4", "9", "7", "12", "0", "5", "undefined", "144", "1", "60"]

report = evaluate_predictions(dataset.examples, predictions)


# 3. DISPLAY THE RESULTS
print(f"Dataset: {dataset.name}")
print(f"Dataset version: {dataset.version}")
print(f"Accuracy: {report.accuracy:.1%}")
print(f"Correct: {report.correct}/{report.total}")

for result in report.results:
    status = "PASS" if result.correct else "FAIL"

    print(
        f"[{status}] [{result.id}] {result.input} | "
        f"Expected: {result.expected} | "
        f"Predicted: {result.predicted}"
    )
