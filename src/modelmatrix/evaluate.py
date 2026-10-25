from dataclasses import dataclass
from typing import Sequence

# ==========================================
# 1. IMMUTABLE DATA SCHEMAS (DATACLASSES)
# ==========================================

# Using frozen=True ensures that once an evaluation artifact is created,
# it cannot be altered by other parts of the platform, preserving the audit trail.

@dataclass(frozen=True)
class EvaluationExample:
    """One test case with an input and its expected answer."""

    id: str
    input: str
    expected: str

@dataclass(frozen=True)
class ExampleResult:
    """The outcome of evaluating one prediction."""

    id: str
    input: str
    expected: str
    predicted: str
    correct: bool

@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics and individual evaluation outcomes."""

    total: int
    correct: int
    accuracy: float
    results: tuple[ExampleResult, ...]


# ==========================================
# 2. STRING EVALUATION UTILITIES
# ==========================================

def normalize_answer(answer: str) -> str:
    """Normalize trivial formatting differences before comparison."""
    return answer.strip().casefold()


# ==========================================
# 3. CORE ORCHESTRATION PIPELINE
# ==========================================

def evaluate_predictions(
    examples: Sequence[EvaluationExample],
    predictions: Sequence[str]
) -> EvaluationReport:
    """Compare predictions against expected answers."""

    if len(examples) != len(predictions):
        raise ValueError(
            "The number of predictions must match the number of examples."
        )
    if not examples:
        raise ValueError("Cannot evaluate an empty dataset.")

    results = []
    for example, prediction in zip(examples, predictions):
        correct = (
            normalize_answer(example.expected)
            == normalize_answer(prediction)
        )

        results.append(
            ExampleResult(
                id=example.id,
                input=example.input,
                expected=example.expected,
                predicted=prediction,
                correct=correct
            )
        )
    correct_count = sum(result.correct for result in results)

    return EvaluationReport(
        total=len(results),
        correct=correct_count,
        accuracy=correct_count / len(results),
        results=tuple(results)
    )
