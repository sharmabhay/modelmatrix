import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from modelmatrix.datasets import EvaluationDataset
from modelmatrix.evaluate import EvaluationReport, ExampleResult


# ==========================================
# 1. IMMUTABLE EXPERIMENT SCHEMAS
# ==========================================

# A run artifact records what was evaluated, when it was evaluated,
# and the individual outcomes. Keeping this metadata with the results
# makes experiments easier to audit and reproduce.

@dataclass(frozen=True)
class RunArtifact:
    """A persistent record of one evaluation run."""

    run_id: str
    created_at: str
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    total: int
    correct: int
    accuracy: float
    results: tuple[ExampleResult, ...]

@dataclass(frozen=True)
class RunComparison:
    """A comparison between a baseline and a candidate run."""

    baseline_run_id: str
    candidate_run_id: str
    total: int
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_delta: float
    improvements: tuple[str, ...]
    regressions: tuple[str, ...]


# ==========================================
# 2. CREATE AND PERSIST RUN ARTIFACTS
# ==========================================

def create_run_artifact(dataset: EvaluationDataset, report: EvaluationReport) -> RunArtifact:
    """Combine dataset identity and evaluation results into one run artifact."""

    run_id = uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat()

    return RunArtifact(
        run_id=run_id,
        created_at=created_at,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_sha256=dataset.sha256,
        total=report.total,
        correct=report.correct,
        accuracy=report.accuracy,
        results=report.results
    )

def save_run(artifact: RunArtifact, output_dir: str | Path) -> Path:
    """Serialize a run to JSON without overwriting earlier runs."""

    output_path = Path(output_dir)

    # Create the output directory if it doesn't exist.
    output_path.mkdir(parents=True, exist_ok=True)

    # Each run gets its own file, preserving previous experiment records.
    file_path = output_path / f"{artifact.run_id}.json"

    with file_path.open("x", encoding="utf-8") as file:
        json.dump(asdict(artifact), file, ensure_ascii=False, indent=2)

    return file_path


# ==========================================
# 3. LOAD AND VALIDATE SAVED RUNS
# ==========================================

def load_run(path: str | Path) -> RunArtifact:
    """Load a saved run and validates its basic structure."""

    run_path = Path(path)
    try:
        with run_path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in run artifact '{run_path}': {error}") from error

    if not isinstance(raw_data, dict):
        raise ValueError("Run artifact must be a JSON object.")

    raw_results = raw_data.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("'results' must be a JSON array.")

    try:
        # Reconstruct typed result objects rather than leaving nested dictionaries
        # inside the loaded experiment.
        results = tuple(ExampleResult(**result) for result in raw_results)

        artifact = RunArtifact(
            run_id=raw_data["run_id"],
            created_at=raw_data["created_at"],
            dataset_name=raw_data["dataset_name"],
            dataset_version=raw_data["dataset_version"],
            dataset_sha256=raw_data["dataset_sha256"],
            total=raw_data["total"],
            correct=raw_data["correct"],
            accuracy=raw_data["accuracy"],
            results=results
        )
    except (TypeError, KeyError) as error:
        raise ValueError(f"Run artifact '{run_path}' has an invalid structure") from error

    # Check that the aggregate metric agree with the saved outcomes which catches
    # inconsistent artifacts before comparing experiments.
    actual_correct = sum(result.correct for result in artifact.results)

    if artifact.total != len(artifact.results):
        raise ValueError("Run total does not match the number of results.")
    if artifact.correct != actual_correct:
        raise ValueError("Run correct count does not match its results.")

    expected_accuracy = artifact.correct / artifact.total if artifact.total else 0.0

    if (artifact.total == 0 or not isinstance(artifact.accuracy, (float, int)) or abs(artifact.accuracy - expected_accuracy) > 1e-9):
        raise ValueError("Run accuracy is inconsistent with its results.")

    return artifact


# ==========================================
# 4. COMPARE EXPERIMENT RUNS
# ==========================================

def compare_runs(baseline: RunArtifact, candidate: RunArtifact) -> RunComparison:
    """Compare two runs on the exact same dataset."""

    # A score comparison is only meaningful when both runs use the same dataset
    # artifact and the same set of example IDs.
    if baseline.dataset_sha256 != candidate.dataset_sha256:
        raise ValueError("Cannot compare runs with different dataset fingerprints.")

    baseline_ids = tuple(result.id for result in baseline.results)
    candidate_ids = tuple(result.id for result in candidate.results)
    if baseline_ids != candidate_ids:
        raise ValueError("Cannot compare runs with different example IDs or ordering.")

    # Identify cases that changed from wrong to correct and vice versa.
    improvements = []
    regressions = []
    for old, new in zip(baseline.results, candidate.results):
        if not old.correct and new.correct:
            improvements.append(new.id)
        elif old.correct and not new.correct:
            regressions.append(new.id)

    return RunComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        total=baseline.total,
        baseline_accuracy=baseline.accuracy,
        candidate_accuracy=candidate.accuracy,
        accuracy_delta=candidate.accuracy - baseline.accuracy,
        improvements=tuple(improvements),
        regressions=tuple(regressions)
    )
