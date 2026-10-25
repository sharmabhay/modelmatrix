import json
import pytest

from modelmatrix.datasets import load_dataset

def valid_dataset():
    return {
        "name": "arithmetic",
        "version": "1.0.0",
        "description": "Basic arithmetic benchmark.",
        "examples": [
            {
                "id": "addition_001",
                "input": "What is 2 + 2?",
                "expected": "4",
            },
            {
                "id": "addition_002",
                "input": "What is 3 + 5?",
                "expected": "8",
            },
        ],
    }

def write_dataset(tmp_path, data):
    """Write test data to a temporary JSON file."""

    path = tmp_path / "dataset.json"

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    return path

def test_loads_dataset_and_metadata(tmp_path):
    path = write_dataset(tmp_path, valid_dataset())

    dataset = load_dataset(path)

    assert dataset.name == "arithmetic"
    assert dataset.version == "1.0.0"
    assert len(dataset.examples) == 2
    assert dataset.examples[0].id == "addition_001"
    assert dataset.examples[0].expected == "4"

def test_rejects_duplicate_example_ids(tmp_path):
    data = valid_dataset()
    data["examples"][1]["id"] = "addition_001"

    path = write_dataset(tmp_path, data)

    with pytest.raises(ValueError, match="Duplicate example ID"):
        load_dataset(path)

def test_rejects_empty_examples(tmp_path):
    data = valid_dataset()
    data["examples"] = []

    path = write_dataset(tmp_path, data)

    with pytest.raises(ValueError, match="non-empty JSON array"):
        load_dataset(path)

def test_rejects_missing_expected_answer(tmp_path):
    data = valid_dataset()
    del data["examples"][0]["expected"]

    path = write_dataset(tmp_path, data)

    with pytest.raises(ValueError, match="examples\\[0\\]\\.expected"):
        load_dataset(path)

def test_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_dataset(path)

def test_rejects_empty_input(tmp_path):
    data = valid_dataset()
    data["examples"][0]["input"] = "   "

    path = write_dataset(tmp_path, data)

    with pytest.raises(ValueError, match="examples\\[0\\]\\.input"):
        load_dataset(path)
