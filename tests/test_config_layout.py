from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from cgdit.rl.config import load_rl_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "conf"


def _compose_model(option: str):
    with initialize_config_dir(
        config_dir=str(CONFIG_ROOT), version_base=None
    ):
        return compose(
            config_name="default",
            overrides=[f"model={option}"],
        ).model


def test_all_diffusion_experiments_compose_shared_components():
    experiments = sorted(
        (CONFIG_ROOT / "model/experiments").glob("*.yaml")
    )

    assert len(experiments) == 28
    for path in experiments:
        model = _compose_model(f"experiments/{path.stem}")
        assert model._target_.startswith("cgdit.pl_modules.")
        assert model.decoder._target_.endswith("CSPNet")
        assert model.beta_scheduler.scheduler_mode == "cosine"
        assert model.sigma_scheduler.sigma_begin == pytest.approx(0.005)
        if path.stem != "diffusion_w_type":
            assert model.d3pm_scheduler.kind == "standard"


@pytest.mark.parametrize(
    ("option", "target"),
    [
        (
            "property_predictors/m3gnet/regression",
            "cgdit.prop_models.gnn_models.m3gnet.M3GNetSurrogate",
        ),
        (
            "property_predictors/m3gnet/tc_regressor",
            "cgdit.prop_models.gnn_models.m3gnet.M3GNetSurrogate",
        ),
        (
            "property_predictors/diffusion_cspnet/tc_regressor",
            "cgdit.prop_models.diffusion_backbone.DiffusionBackboneRegressor",
        ),
        (
            "property_predictors/diffusion_cspnet/tc_classifier",
            "cgdit.prop_models.tc_classifier.DiffusionBackboneBinaryClassifier",
        ),
    ],
)
def test_property_predictor_configs_compose(option, target):
    model = _compose_model(option)

    assert model._target_ == target
    if "diffusion_cspnet" in option:
        assert model.decoder.latent_dim == 256
        assert model.decoder.pred_scalar is True


def test_all_rl_experiments_resolve_required_components():
    required = {
        "model_path",
        "reward_contract",
        "reward_registry",
        "property",
        "algorithm",
        "output_root",
        "learning_rate",
        "replay_transitions",
        "seed",
    }
    experiments = sorted((CONFIG_ROOT / "rl/experiments").glob("*.yaml"))

    assert len(experiments) == 21
    for path in experiments:
        config = load_rl_config(path)
        assert required.issubset(config.keys())
        assert "includes" not in config


def test_rl_experiment_values_override_included_components(tmp_path):
    component = tmp_path / "component.yaml"
    component.write_text("updates: 8\nseed: 42\n", encoding="utf-8")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "includes:\n"
        "  - component.yaml\n"
        "updates: 200\n",
        encoding="utf-8",
    )

    config = load_rl_config(experiment)

    assert config.updates == 200
    assert config.seed == 42


def test_rl_config_rejects_include_cycles(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("includes: [second.yaml]\n", encoding="utf-8")
    second.write_text("includes: [first.yaml]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Cyclic RL config include"):
        load_rl_config(first)


def test_diffusion_configuration_layout_is_flat():
    diffusion_root = CONFIG_ROOT / "model/diffusion"

    assert not (CONFIG_ROOT / "model/components").exists()
    assert not (diffusion_root / "diffusion").exists()
    assert {path.name for path in diffusion_root.glob("*.yaml")} == {
        "standard.yaml", "multimodal.yaml", "with_type.yaml"
    }


def test_property_predictor_configuration_layout():
    predictor_root = CONFIG_ROOT / "model/property_predictors"

    assert not (predictor_root / "components").exists()
    assert not (predictor_root / "diffusion_backbone").exists()
    assert not (predictor_root / "tc_experiments").exists()
    assert not (predictor_root / "legacy").exists()
    assert {
        path.name for path in (predictor_root / "backbone").glob("*.yaml")
    } == {"diffusion_cspnet.yaml", "m3gnet.yaml"}
    assert {
        path.name
        for path in (predictor_root / "diffusion_cspnet").glob("*.yaml")
    } == {"tc_regressor.yaml", "tc_classifier.yaml"}
    assert {
        path.name for path in (predictor_root / "m3gnet").glob("*.yaml")
    } == {"regression.yaml", "tc_regressor.yaml"}


def test_legacy_flat_config_directories_are_removed():
    model_root = CONFIG_ROOT / "model"

    for name in (
        "beta_scheduler",
        "sigma_scheduler",
        "d3pm_scheduler",
        "decoder",
        "prop_models",
    ):
        assert not (model_root / name).exists()
    assert not list(model_root.glob("exp_*.yaml"))
    assert not list((CONFIG_ROOT / "rl").glob("ppo_*.yaml"))
    assert not list((CONFIG_ROOT / "rl").glob("grpo_*.yaml"))
