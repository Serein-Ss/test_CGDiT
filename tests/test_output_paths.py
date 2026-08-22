import json
from pathlib import Path

from cgdit.common.output_paths import (
    evaluation_output_path,
    generation_output_path,
    model_root_from_generation,
    record_evaluation_metric,
    record_generated_structure,
    resolve_evaluation_output_path,
    resolve_generation_output_path,
)
from scripts.cli.tools.migrate_output_layout import (
    apply_migration,
    build_migration_plan,
    verify_migration,
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


def test_resolve_generation_output_path_accepts_legacy_result(tmp_path):
    legacy = tmp_path / "eval_gen_template_uncond_n4096_seed42.pt"
    legacy.write_bytes(b"result")

    resolved = resolve_generation_output_path(
        tmp_path, "template_uncond_n4096_seed42"
    )

    assert resolved == legacy
    assert model_root_from_generation(legacy) == tmp_path


def test_resolve_evaluation_output_path_accepts_legacy_result(tmp_path):
    generation = tmp_path / "eval_gen_template_uncond_n4096_seed42.pt"
    generation.write_bytes(b"result")
    legacy = tmp_path / "eval_metrics_gen_template_uncond_n4096_seed42.json"
    legacy.write_text("{}", encoding="utf-8")

    resolved = resolve_evaluation_output_path(
        generation,
        "structural_metrics",
        legacy.name,
    )

    assert resolved == legacy


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


def test_legacy_output_migration_keeps_verified_backup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_root = Path("output/singlerun/date/model")
    model_root.mkdir(parents=True)
    (model_root / "hparams.yaml").write_text("model: {}\n", encoding="utf-8")
    legacy_generation = model_root / "eval_gen_template_uncond_n4096_seed42.pt"
    legacy_metrics = model_root / "eval_metrics_gen_template_uncond_n4096_seed42.json"
    legacy_generation.write_bytes(b"generated")
    legacy_metrics.write_text("{}", encoding="utf-8")
    backup = Path("output/backup/singlerun")
    manifest = Path("output/migration.json")

    items = build_migration_plan("output/singlerun", backup)
    apply_migration(items, manifest)

    canonical_generation = generation_output_path(
        model_root, "template_uncond_n4096_seed42"
    )
    canonical_metrics = evaluation_output_path(
        canonical_generation,
        "structural_metrics",
        legacy_metrics.name,
    )
    assert len(items) == 2
    assert not legacy_generation.exists()
    assert not legacy_metrics.exists()
    assert canonical_generation.read_bytes() == b"generated"
    assert canonical_metrics.read_text(encoding="utf-8") == "{}"
    assert (backup / "date/model" / legacy_generation.name).read_bytes() == b"generated"
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "complete"
    assert verify_migration(manifest) == (2, len(b"generated") + len(b"{}"))
