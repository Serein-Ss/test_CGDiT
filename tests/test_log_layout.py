from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (
    PROJECT_ROOT,
    PROJECT_ROOT / "submit",
    PROJECT_ROOT / "submit_python",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "cgdit",
)
RUNTIME_SUFFIXES = {".py", ".sh", ".slurm"}
LOG_CATEGORIES = {
    "generative_model_training",
    "structure_generation",
    "metric_evaluation",
    "property_predictor_training",
    "reinforcement_learning",
    "tests",
    "other",
}
LEGACY_LOG_PATHS = {
    "logs/all_generation_evaluation",
    "logs/mp20_predictors",
    "logs/mp20_predictors_multiseed",
    "logs/mp20_remaining",
    "logs/mp20_runs",
    "logs/remote_rl",
    "logs/rl/",
    "logs/rl_generation",
    "logs/magndata_tc",
    "logs/magndata_tc_moe",
}


def _runtime_files():
    seen = set()
    for root in RUNTIME_ROOTS:
        candidates = (root.iterdir() if root == PROJECT_ROOT else root.rglob("*"))
        for path in candidates:
            if path.is_file() and path.suffix in RUNTIME_SUFFIXES and path not in seen:
                seen.add(path)
                yield path


def test_runtime_code_does_not_write_to_legacy_log_paths():
    violations = {}
    for path in _runtime_files():
        text = path.read_text(encoding="utf-8")
        matches = sorted(marker for marker in LEGACY_LOG_PATHS if marker in text)
        if matches:
            violations[str(path.relative_to(PROJECT_ROOT))] = matches
    assert not violations


def test_all_log_categories_are_used_by_runtime_code():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _runtime_files()
    )
    missing = sorted(
        category for category in LOG_CATEGORIES
        if f"logs/{category}" not in text
    )
    assert not missing
