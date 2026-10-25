import pytest

from modelmatrix.evaluate import (
    EvaluationExample,
    evaluate_predictions,
    normalize_answer,
)

def test_accuracy_is_calculated_correctly():
    examples = [
        EvaluationExample("q1", "2 + 2", "4"),
        EvaluationExample("q2", "3 + 5", "8"),
        EvaluationExample("q3", "10 - 3", "7")
    ]
    predictions = ["4", "9", "7"]

    report = evaluate_predictions(examples, predictions)

    assert report.total == 3
    assert report.correct == 2
    assert report.accuracy == pytest.approx(2 / 3)

    assert [result.correct for result in report.results] == [
        True, False, True
    ]

def test_normalization_ignores_case_and_outer_whitespace():
    assert normalize_answer("  HELLO  ") == "hello"

def test_mismatched_prediction_count_raises_error():
    examples = [EvaluationExample("q1", "2 + 2", "4")]

    with pytest.raises(ValueError):
        evaluate_predictions(examples, [])

def test_empty_dataset_raises_error():
    with pytest.raises(ValueError):
        evaluate_predictions([], [])

def test_result_preserves_example_id():
    examples = [EvaluationExample("addition_001", "2 + 2", "4")]

    report = evaluate_predictions(examples, ["4"])

    assert report.results[0].id == "addition_001"
