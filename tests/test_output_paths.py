import json
from pathlib import Path

from cgdit.common.output_paths import (
    evaluation_output_path,
    generation_output_path,
    record_evaluation_metric,
    record_generated_structure,
)


def test_generation_output_path_classifies_generation_tasks(tmp_path):
    model_root = tmp_path / "model"

    assert generation_output_path(
        model_root, "template_fe_m1p5_n4096_seed42"
    ) == (
        model_root
        / "generated_structures/formal/template/conditional"
        / "eval_gen_template_fe_m1p5_n4096_seed42.pt"
    )
    assert generation_output_path(
        model_root, "abinitio_empirical_uncond_n4096_seed42"
    ) == (
        model_root
        / "generated_structures/formal/abinitio_empirical/unconditional"
        / "eval_gen_abinitio_empirical_uncond_n4096_seed42.pt"
    )
    assert generation_output_path(
        model_root, "pilot_template_uncond_seed42"
    ) == (
        model_root
        / "generated_structures/pilot/template/unconditional"
        / "eval_gen_pilot_template_uncond_seed42.pt"
    )
    assert generation_output_path(model_root, "abinitio_fix_smoke") == (
        model_root
        / "generated_structures/test/abinitio/unconditional"
        / "eval_gen_abinitio_fix_smoke.pt"
    )


def test_evaluation_output_path_uses_model_evaluation_directory(tmp_path):
    model_root = tmp_path / "model"
    generation = generation_output_path(
        model_root, "template_uncond_n4096_seed42"
    )

    assert evaluation_output_path(
        generation, "structural_metrics", "metrics.json"
    ) == model_root / "evaluations/structural_metrics/metrics.json"


def test_source_manifest_records_sources_without_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_root = Path("output/singlerun/date/model")
    generation = generation_output_path(
        model_root, "template_uncond_n4096_seed42"
    )
    metric = evaluation_output_path(
        generation, "structural_metrics", "metrics.json"
    )

    record_generated_structure(generation)
    record_generated_structure(generation)
    record_evaluation_metric("structural_metrics", metric, generation)
    record_evaluation_metric("structural_metrics", metric, generation)

    manifest = json.loads(
        (model_root / "evaluations/source_manifest.json").read_text()
    )
    assert manifest["model_directory"] == str(model_root)
    assert len(manifest["generated_structures"]) == 1
    assert len(manifest["evaluation_metrics"]) == 1
    assert manifest["evaluation_metrics"][0]["source_generation"] == str(
        generation
    )
