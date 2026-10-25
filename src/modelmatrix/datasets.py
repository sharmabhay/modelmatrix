import json
from dataclasses import dataclass
from pathlib import Path

from modelmatrix.evaluate import EvaluationExample

# ==========================================
# 1. DATASET METADATA SCHEMA
# ==========================================

# Using frozen=True ensures that once an evaluation artifact is created,
# it cannot be altered by other parts of the platform, preserving the audit trail.

@dataclass(frozen=True)
class EvaluationDataset:
    """A named, versioned collection of evaluation examples."""

    name: str
    version: str
    description: str
    examples: tuple[EvaluationExample, ...]


# ==========================================
# 2. VALIDATION UTILITIES
# ==========================================

def _require_nonempty_string(value: object, field_name: str) -> str:
    """Validate a required string field and return its value."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"'{field_name}' must be a non-empty string."
        )

    return value

def _require_object(value: object, location: str) -> dict:
    """Ensure a JSON value is an object before accessing its fields."""

    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object.")

    return value


# ==========================================
# 3. DATASET LOADING PIPELINE
# ==========================================

def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate an evaluation dataset from a JSON file."""

    dataset_path = Path(path)
    try:
        with dataset_path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {dataset_path}: {error}") from error

    raw_data = _require_object(raw_data, "Dataset")

    name = _require_nonempty_string(raw_data.get("name"), "name")
    version = _require_nonempty_string(raw_data.get("version"), "version")
    description = _require_nonempty_string(raw_data.get("description"), "description")

    raw_examples = raw_data.get("examples")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("'examples' must be a non-empty JSON array.")

    examples = []
    seen_ids = set()
    for index, raw_example in enumerate(raw_examples):
        location = f"Example {index}"
        raw_example = _require_object(raw_example, location)

        example_id = _require_nonempty_string(raw_example.get("id"), f"{location} 'id'")
        input_text = _require_nonempty_string(raw_example.get("input"), f"{location} 'input'")
        expected = _require_nonempty_string(raw_example.get("expected"), f"{location} 'expected'")

        if example_id in seen_ids:
            raise ValueError(f"Duplicate example ID '{example_id}' found in dataset.")
        seen_ids.add(example_id)

        examples.append(
            EvaluationExample(
                id=example_id,
                input=input_text,
                expected=expected
            )
        )

    return EvaluationDataset(
        name=name,
        version=version,
        description=description,
        examples=tuple(examples)
    )
