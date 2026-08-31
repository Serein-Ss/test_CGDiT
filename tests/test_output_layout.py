import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (
    PROJECT_ROOT,
    PROJECT_ROOT / "conf",
    PROJECT_ROOT / "submit",
    PROJECT_ROOT / "submit_python",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "cgdit",
)
RUNTIME_SUFFIXES = {".py", ".sh", ".slurm", ".yaml"}
LEGACY_OUTPUT_PATHS = {
    "output/rl_finetune",
    "output/rl_generation",
    "output/rl_diagnostics",
    "output/rl_analysis",
    "output/compatibility",
}


def _runtime_files():
    seen = set()
    for root in RUNTIME_ROOTS:
        candidates = root.iterdir() if root == PROJECT_ROOT else root.rglob("*")
        for path in candidates:
            if path.is_file() and path.suffix in RUNTIME_SUFFIXES and path not in seen:
                seen.add(path)
                yield path


def test_runtime_code_does_not_reference_legacy_output_roots():
    violations = {}
    for path in _runtime_files():
        text = path.read_text(encoding="utf-8")
        matches = sorted(marker for marker in LEGACY_OUTPUT_PATHS if marker in text)
        if matches:
            violations[str(path.relative_to(PROJECT_ROOT))] = matches
    assert not violations


def test_rl_output_helper_builds_timestamped_paths():
    command = (
        f'source "{PROJECT_ROOT}/submit_python/rl_output_layout.sh"; '
        f'rl_run_root "{PROJECT_ROOT}" reinforcement_learning '
        'grpo-fe 20260831-123456'
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == str(
        PROJECT_ROOT
        / "output/reinforcement_learning/2026-08-31/12-34-56-grpo-fe"
    )


def test_output_directory_contains_no_runtime_logs():
    root = PROJECT_ROOT / "output"
    misplaced = []
    if root.is_dir():
        misplaced.extend(root.rglob("*.log"))
        misplaced.extend(root.rglob("*.out"))
        misplaced.extend(root.rglob("*.err"))
    assert not misplaced
