import json
import pytest
from dataclasses import replace

from modelmatrix.datasets import load_dataset
from modelmatrix.evaluate import (
    EvaluationExample,
    evaluate_predictions,
)
from modelmatrix.experiments import (
    compare_runs,
    create_run_artifact,
    load_run,
    save_run,
)

def make_dataset(tmp_path):
    data = {
        "name": "arithmetic",
        "version": "1.0.0",
        "description": "Arithmetic experiment tests.",
        "examples": [
            {
                "id": "q1",
                "input": "What is 2 + 2?",
                "expected": "4",
            },
            {
                "id": "q2",
                "input": "What is 3 + 5?",
                "expected": "8",
            },
        ],
    }

    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    return load_dataset(path)

def make_run(dataset, predictions):
    """Evaluate predictions and create a run artifact."""

    report = evaluate_predictions(
        dataset.examples,
        predictions,
    )

    return create_run_artifact(dataset, report)

def test_run_survives_save_and_load(tmp_path):
    dataset = make_dataset(tmp_path)
    artifact = make_run(dataset, ["4", "9"])

    path = save_run(artifact, tmp_path / "runs")
    loaded = load_run(path)

    assert loaded.run_id == artifact.run_id
    assert loaded.dataset_sha256 == dataset.sha256
    assert loaded.accuracy == artifact.accuracy
    assert loaded.results == artifact.results

def test_run_timestamp_is_in_utc(tmp_path):
    dataset = make_dataset(tmp_path)
    artifact = make_run(dataset, ["4", "8"])

    assert artifact.created_at.endswith("+00:00")

def test_comparison_detects_improvements_and_regressions(tmp_path):
    dataset = make_dataset(tmp_path)

    baseline = make_run(dataset, ["4", "9"])
    candidate = make_run(dataset, ["9", "8"])

    comparison = compare_runs(baseline, candidate)

    assert comparison.baseline_accuracy == pytest.approx(0.5)
    assert comparison.candidate_accuracy == pytest.approx(0.5)
    assert comparison.accuracy_delta == pytest.approx(0.0)
    assert comparison.improvements == ("q2",)
    assert comparison.regressions == ("q1",)

def test_comparison_rejects_different_dataset_hashes(tmp_path):
    dataset = make_dataset(tmp_path)

    baseline = make_run(dataset, ["4", "8"])
    different_dataset = replace(
        dataset,
        sha256="different-fingerprint",
    )
    candidate = make_run(different_dataset, ["4", "8"])

    with pytest.raises(ValueError, match="different dataset fingerprints"):
        compare_runs(baseline, candidate)

def test_load_run_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{broken json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_run(path)

def test_load_run_rejects_inconsistent_accuracy(tmp_path):
    dataset = make_dataset(tmp_path)
    artifact = make_run(dataset, ["4", "9"])

    corrupted = replace(artifact, accuracy=1.0)
    path = save_run(corrupted, tmp_path / "runs")

    with pytest.raises(ValueError, match="accuracy is inconsistent"):
        load_run(path)
