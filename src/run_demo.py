from modelmatrix.evaluate import (
    EvaluationExample,
    evaluate_predictions
)


examples = [
    EvaluationExample("What is 2 + 2?", "4"),
    EvaluationExample("What is 3 + 5?", "8"),
    EvaluationExample("What is 10 - 3?", "7"),
    EvaluationExample("What is 4 * 3?", "12"),
    EvaluationExample("What is 0 * 7?", "0"),
    EvaluationExample("What is 19 / 4?", "4.75"),
    EvaluationExample("What is 0 / 0?", "undefined"),
    EvaluationExample("What is 12 ^ 2?", "144"),
    EvaluationExample("What is 0 ^ 0?", "1"),
    EvaluationExample("What is 5 factorial?", "120")
]

# These are simulated predictions, not real model outputs.
predictions = ["4", "9", "7", "12", "0", "5", "undefined", "144", "1", "60"]

report = evaluate_predictions(examples, predictions)

print(f"Accuracy: {report.accuracy:.1%}")
print(f"Correct: {report.correct}/{report.total}")

for result in report.results:
    status = "PASS" if result.correct else "FAIL"
    print(
        f"[{status}] {result.input} | "
        f"Expected: {result.expected} | "
        f"Predicted: {result.predicted}"
    )
