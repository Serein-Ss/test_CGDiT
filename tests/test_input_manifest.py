import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts.cli.data.verify_inputs import verify


def test_input_manifest_detects_changed_and_missing_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    sample = data / "sample.csv"
    sample.write_bytes(b"cif,value\nexample,1\n")
    item = {
        "path": "data/sample.csv",
        "bytes": sample.stat().st_size,
        "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
    }
    (data / "manifest.json").write_text(json.dumps({"files": [item]}))
    assert verify(tmp_path) == []
    sample.write_bytes(b"changed")
    assert verify(tmp_path) == ["Changed: data/sample.csv"]
    sample.unlink()
    assert verify(tmp_path) == ["Missing: data/sample.csv"]


def test_predictor_launcher_accepts_csv_only_clean_clone(tmp_path):
    data = tmp_path / "data/mp_20"
    data.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (data / f"{split}.csv").write_text("material_id,cif\nexample,placeholder\n")
    executables = tmp_path / "bin"
    executables.mkdir()
    python = executables / "python"
    # Exercise shell preflight and all three dispatches without running training.
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PROJECT_ROOT/calls.txt"\n')
    python.chmod(0o755)
    env = dict(os.environ, PROJECT_ROOT=str(tmp_path), SKIP_EXISTING="0")
    env["PATH"] = str(executables) + os.pathsep + env["PATH"]
    script = Path(__file__).resolve().parents[1] / "submit_python/run_all_mp20_predictors.sh"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls.txt").read_text()
    assert calls.count("cgdit/run.py") == 3
    assert not list(data.glob("*.pt"))
