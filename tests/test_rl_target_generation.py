from pathlib import Path

from cgdit.common.output_paths import _generation_parts
from cgdit.evaluation.generated_properties import discover_formal_generation_files
from cgdit.prop_models.diffusion_backbone import resolve_checkpoint


def test_rl_generation_labels_use_canonical_output_layout():
    for target_label in ("fe_m1p5", "bg_2"):
        assert _generation_parts(
            f"abinitio_empirical_rl_{target_label}_n4096_seed42"
        ) == ("formal", "abinitio_empirical", "conditional")
        assert _generation_parts(f"template_rl_{target_label}_n4096_seed42") == (
            "formal",
            "template",
            "conditional",
        )


def test_property_evaluator_discovers_both_rl_generation_modes(tmp_path: Path):
    abinitio = tmp_path / "eval_gen_abinitio_empirical_rl_fe_m1p5_n4096_seed42.pt"
    template = tmp_path / "eval_gen_template_rl_fe_m1p5_n4096_seed42.pt"
    abinitio.write_bytes(b"x")
    template.write_bytes(b"x")
    assert discover_formal_generation_files(tmp_path) == [abinitio, template]


def test_checkpoint_resolution_prefers_largest_step(tmp_path: Path):
    first = tmp_path / "epoch=19-step=19.ckpt"
    last = tmp_path / "epoch=39-step=39.ckpt"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    assert resolve_checkpoint(tmp_path) == last.resolve()
