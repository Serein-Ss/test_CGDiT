import json

import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Lattice, Structure

from scripts.result_figures.crystalpirl_article_blueprint import plot_blueprints
from scripts.result_figures.crystalpirl_article_blueprint.build_fig3_quality_metrics import (
    _metric_row,
    _mp_hull_distances,
)
from scripts.result_figures.crystalpirl_article_blueprint.build_fig4_source_data import (
    _valid_joint_target_mask,
)


def test_available_quality_metrics_uses_structural_and_property_results(tmp_path, monkeypatch):
    metric_paths = []
    for index, comp_valid in enumerate([0.75, 0.80]):
        metric_path = tmp_path / f"metrics_{index}.json"
        metric_path.write_text(
            json.dumps(
                {
                    "comp_valid": comp_valid,
                    "struct_valid": comp_valid + 0.10,
                    "valid": comp_valid - 0.01,
                }
            )
        )
        metric_paths.append(metric_path)

    monkeypatch.setattr(plot_blueprints, "BASE_STRUCTURAL_METRICS", metric_paths[0])
    monkeypatch.setattr(plot_blueprints, "JOINT_CFG_STRUCTURAL_METRICS", metric_paths[1])
    monkeypatch.setattr(
        plot_blueprints, "FIG3_QUALITY_METRICS", tmp_path / "missing.csv"
    )

    result = plot_blueprints._available_quality_metrics()

    assert result["method"].tolist() == ["Base", "CFG"]
    np.testing.assert_allclose(result["compositional_validity"], [75.0, 80.0])
    np.testing.assert_allclose(result["structural_validity"], [85.0, 90.0])
    assert result["mp_hull_stability"].isna().all()



def test_available_quality_metrics_adds_full_set_quality_results(tmp_path, monkeypatch):
    metrics_path = tmp_path / "quality.csv"
    pd.DataFrame(
        {
            "method": ["Base", "CFG"],
            "sun": [0.21, 0.32],
            "uniqueness": [0.81, 0.82],
            "novelty_among_unique": [0.71, 0.72],
            "mp_hull_stability": [0.61, 0.62],
            "novelty_reference": ["materials_project", "materials_project"],
        }
    ).to_csv(metrics_path, index=False)
    monkeypatch.setattr(plot_blueprints, "FIG3_QUALITY_METRICS", metrics_path)

    result = plot_blueprints._available_quality_metrics()

    np.testing.assert_allclose(result["sun"], [21.0, 32.0])
    np.testing.assert_allclose(result["uniqueness"], [81.0, 82.0])
    np.testing.assert_allclose(result["novelty_among_unique"], [71.0, 72.0])
    np.testing.assert_allclose(result["mp_hull_stability"], [61.0, 62.0])

def test_plotting_values_excludes_band_gaps_above_ten():
    values = np.asarray([0.0, 2.0, 10.0, 10.01, np.nan])

    result = plot_blueprints._plotting_values(values, "band_gap")

    np.testing.assert_allclose(result, [0.0, 2.0, 10.0])


def test_plotting_values_clips_negative_band_gaps_to_zero():
    values = np.asarray([-2.0, -0.1, 1.0])

    result = plot_blueprints._plotting_values(values, "band_gap")

    np.testing.assert_allclose(result, [0.0, 0.0, 1.0])


def test_highest_density_threshold_encloses_requested_mass():
    density = np.asarray([[4.0, 3.0], [2.0, 1.0]])

    threshold = plot_blueprints._highest_density_threshold(
        density, mass_fraction=0.60
    )

    assert threshold == 3.0
    assert density[density >= threshold].sum() / density.sum() >= 0.60


def test_mp_hull_distance_compares_formation_energies_on_the_same_scale():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Li", "Li", "O"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.25, 0.25, 0.25]],
    )
    reference = {
        ("Li",): [PDEntry("Li", 0.0)],
        ("O",): [PDEntry("O", 0.0)],
        ("Li", "O"): [PDEntry("Li2O", -3.0)],
    }

    hull, e_above_hull = _mp_hull_distances(
        [structure],
        valid=np.array([True]),
        formation_energy=np.array([-0.8]),
        reference=reference,
    )

    np.testing.assert_allclose(hull, [-1.0])
    np.testing.assert_allclose(e_above_hull, [0.2])


def test_metric_row_uses_declared_quality_denominators():
    result = _metric_row(
        method="Base",
        valid=np.array([True, True, True, False]),
        stable=np.array([True, False, True, False]),
        representatives=np.array([0, 2]),
        novel=np.array([True, False, True, False]),
        composition_valid=np.array([True, True, True, True]),
        structure_valid=np.array([True, True, True, False]),
        novelty_reference="mp_20_train",
    )

    assert result["uniqueness"] == 2 / 3
    assert result["novelty_among_unique"] == 1.0
    assert result["sun"] == 2 / 4
    assert result["mp_hull_stability"] == 2 / 4



def test_valid_joint_target_mask_includes_validity_and_both_targets():
    predictions = pd.DataFrame(
        {
            "predicted_formation_energy_per_atom": [-1.5, -1.5, -1.0],
            "predicted_band_gap": [2.0, 2.0, 2.0],
        }
    )

    result = _valid_joint_target_mask(
        np.array([True, False, True]), predictions
    )

    np.testing.assert_array_equal(result, [True, False, False])


def test_interim_reward_shift_compares_first_and_last_eight_updates():
    records = [
        {"reward": {"reward_mean": value}}
        for value in ([0.0] * 8 + [1.0] * 8)
    ]

    result = plot_blueprints._interim_reward_shift(records, resamples=100)

    np.testing.assert_allclose(result, [1.0, 1.0, 1.0])


def test_rl_records_uses_resume_record_for_duplicate_step(tmp_path, monkeypatch):
    initial = tmp_path / "initial.log"
    resume = tmp_path / "resume.log"
    initial.write_text(
        '\n'.join(
            json.dumps({"step": step, "algorithm": "grpo", "source": "initial"})
            for step in (0, 1)
        )
    )
    resume.write_text(
        '\n'.join(
            json.dumps({"step": step, "algorithm": "grpo", "source": "resume"})
            for step in (1, 2)
        )
    )
    monkeypatch.setattr(plot_blueprints, "RL_LOGS", {"fe": (initial, resume)})

    records = plot_blueprints._rl_records("fe")

    assert [record["step"] for record in records] == [0, 1, 2]
    assert records[1]["source"] == "resume"


def test_interim_paired_evaluations_requires_both_properties(tmp_path, monkeypatch):
    paths = {}
    for task in ("fe", "bg"):
        path = tmp_path / f"{task}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "interim_checkpoint_paired_evaluation",
                    "completed_updates": 40,
                    "evaluation": {"property": task},
                }
            )
        )
        paths[task] = path
    monkeypatch.setattr(plot_blueprints, "INTERIM_PAIRED", paths)

    evaluations, completed_updates = (
        plot_blueprints._interim_paired_evaluations()
    )

    assert set(evaluations) == {"fe", "bg"}
    assert completed_updates == 40
