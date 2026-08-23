from types import SimpleNamespace

import pytest
import torch

from scripts.cli.training.train_crystal_rl import (
    _advantage_summary,
    _blend_decoder,
    _configured_safety_tolerances,
    _evaluation_safety_metrics,
    _grouped_batch,
    _prompt_indices,
    _optimize_policy_epochs,
    _safety_audit,
    _run_output_paths,
    _transition_indices,
    _verified_update_outcome,
)


class _Prompt:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return _Prompt(self.value)


def test_transition_indices_cover_endpoints():
    assert _transition_indices(10, 3) == [0, 4, 9]
    assert _transition_indices(2, 5) == [0, 1]


def test_grouped_batch_repeats_each_prompt(monkeypatch):
    captured = []

    def fake_batch(data):
        captured.extend(item.value for item in data)
        return SimpleNamespace()

    monkeypatch.setattr(
        "scripts.cli.training.train_crystal_rl.Batch.from_data_list", fake_batch
    )
    _, groups = _grouped_batch([_Prompt("a"), _Prompt("b")], 2, 3)

    assert captured == ["a", "a", "a", "b", "b", "b"]
    assert groups.tolist() == [0, 0, 0, 1, 1, 1]


def test_grouped_batch_uses_explicit_prompt_indices(monkeypatch):
    captured = []

    def fake_batch(data):
        captured.extend(item.value for item in data)
        return SimpleNamespace()

    monkeypatch.setattr(
        "scripts.cli.training.train_crystal_rl.Batch.from_data_list", fake_batch
    )
    _grouped_batch(
        [_Prompt("a"), _Prompt("b"), _Prompt("c")],
        2,
        2,
        prompt_indices=[2, 0],
    )

    assert captured == ["c", "c", "a", "a"]


def test_prompt_indices_are_seeded_distinct_and_reproducible():
    train_0 = _prompt_indices(100, 4, seed=42)
    train_1 = _prompt_indices(100, 4, seed=43)
    fixed_probe = _prompt_indices(100, 4, seed=4242)
    holdout_probe = _prompt_indices(100, 4, seed=24242)

    assert train_0 == _prompt_indices(100, 4, seed=42)
    assert len(set(train_0)) == 4
    assert train_0 != train_1
    assert fixed_probe != holdout_probe


def test_prompt_schedule_can_cover_sixteen_unique_training_prompts():
    schedule = _prompt_indices(100, 16, seed=42)

    assert len(schedule) == 16
    assert len(set(schedule)) == 16
    assert schedule == _prompt_indices(100, 16, seed=42)


def test_blend_decoder_applies_decision_scale():
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.decoder.weight.fill_(3.0)
    old_state = {"weight": torch.tensor([[1.0]])}

    _blend_decoder(model, old_state, 0.25)

    assert model.decoder.weight.item() == pytest.approx(1.5)


def test_advantage_summary_keeps_group_diagnostics():
    summary = _advantage_summary(
        torch.tensor([-1.0, 1.0, -2.0, 2.0]),
        torch.tensor([0, 0, 1, 1]),
    )

    assert summary["mean"] == pytest.approx(0.0)
    assert summary["std"] == pytest.approx(1.5811388)
    assert summary["groups"] == {
        "0": {"mean": 0.0, "std": 1.0},
        "1": {"mean": 0.0, "std": 2.0},
    }


def test_rl_test_outputs_separate_model_and_test_results(tmp_path):
    model_dir, results_dir = _run_output_paths(
        tmp_path, "grpo_fe_seed42", "test"
    )

    assert model_dir == tmp_path / "grpo_fe_seed42/model"
    assert results_dir == tmp_path / "grpo_fe_seed42/test_results"


def test_policy_epochs_reuse_one_rollout_after_the_policy_changes():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    class FakeObjective:
        algorithm = "ppo"

        def __call__(self, **_kwargs):
            ratio = parameter.exp()
            return SimpleNamespace(
                loss=-parameter,
                metrics={
                    "approx_kl": -parameter.detach(),
                    "clip_fraction": (ratio.detach() > 1.05).float(),
                    "ratio_mean": ratio.detach(),
                },
            )

    _, _, _, records = _optimize_policy_epochs(
        objective=FakeObjective(),
        model=SimpleNamespace(),
        batch=SimpleNamespace(),
        trajectory=SimpleNamespace(),
        step_lr=1.0e-5,
        rewards=torch.ones(2),
        advantages=torch.tensor([-1.0, 1.0]),
        group_index=torch.tensor([0, 0]),
        transition_indices=[0],
        reference_weight=0.0,
        optimizer=optimizer,
        trainable_parameters=[parameter],
        policy_epochs=3,
    )

    assert len(records) == 3
    assert records[0]["ratio_mean"] == pytest.approx(1.0)
    assert records[1]["ratio_mean"] > 1.0
    assert records[1]["clip_fraction"] == pytest.approx(1.0)


def test_policy_epoch_streams_transition_graphs_before_one_optimizer_step():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    calls = []

    class FakeObjective:
        algorithm = "ppo"

        def __call__(self, **kwargs):
            indices = kwargs["transition_indices"]
            calls.append(indices)
            transition = indices[0]
            target = parameter.new_tensor(0.1 * (transition + 1))
            return SimpleNamespace(
                loss=(parameter - target).square(),
                metrics={
                    "approx_kl": parameter.new_tensor(float(transition)),
                    "clip_fraction": parameter.new_tensor(0.0),
                    "ratio_mean": parameter.new_tensor(float(transition + 1)),
                },
            )

    result, loss, gradient_norm, _ = _optimize_policy_epochs(
        objective=FakeObjective(),
        model=SimpleNamespace(),
        batch=SimpleNamespace(),
        trajectory=SimpleNamespace(),
        step_lr=1.0e-5,
        rewards=torch.ones(2),
        advantages=torch.tensor([-1.0, 1.0]),
        group_index=torch.tensor([0, 0]),
        transition_indices=[0, 1],
        reference_weight=0.0,
        optimizer=optimizer,
        trainable_parameters=[parameter],
        policy_epochs=1,
    )

    assert calls == [[0], [1]]
    assert loss.item() == pytest.approx(0.025)
    assert gradient_norm == pytest.approx(0.3)
    assert parameter.item() == pytest.approx(0.03)
    assert result.metrics["approx_kl"].item() == pytest.approx(0.5)
    assert result.metrics["ratio_mean"].item() == pytest.approx(1.5)


def test_partial_safety_gate_is_diagnostic_only_in_test_mode():
    audit = _safety_audit(
        ["validity", "stability"], ["validity"], run_kind="test"
    )

    assert audit["verification_status"] == "diagnostic_only"
    assert audit["missing"] == ["stability"]
    assert not audit["formal_complete"]


def test_configured_safety_tolerances_are_metric_specific():
    contract = SimpleNamespace(
        closed_loop={
            "safety_drop_tolerances": {
                "validity": 0.01,
                "diversity": 0.05,
            }
        }
    )

    assert _configured_safety_tolerances(
        contract, ["validity", "diversity"]
    ) == {"validity": 0.01, "diversity": 0.05}


def test_partial_safety_gate_blocks_formal_training():
    with pytest.raises(RuntimeError, match="stability"):
        _safety_audit(
            ["validity", "stability"], ["validity"], run_kind="train"
        )


def test_evaluation_safety_metrics_keeps_only_aligned_required_values():
    evaluation = SimpleNamespace(
        valid=torch.tensor([True, False]),
        safety_metrics={
            "stability": torch.tensor([0.9, 0.8]),
            "uniqueness": torch.tensor([1.0, 0.5]),
        },
    )

    metrics = _evaluation_safety_metrics(
        evaluation, ["validity", "stability", "novelty"]
    )

    assert list(metrics) == ["validity", "stability"]
    assert torch.equal(metrics["validity"], torch.tensor([1.0, 0.0]))


def test_evaluation_safety_metrics_rejects_misaligned_values():
    evaluation = SimpleNamespace(
        valid=torch.tensor([True, False]),
        safety_metrics={"stability": torch.ones(3)},
    )

    with pytest.raises(ValueError, match="stability"):
        _evaluation_safety_metrics(evaluation, ["validity", "stability"])


def test_incomplete_safety_cannot_advance_verified_policy():
    action, verified, reason = _verified_update_outcome(
        probe_action="accept",
        safety_complete=False,
        holdout_action=None,
    )

    assert action == "reject"
    assert not verified
    assert reason == "incomplete_safety"


def test_failed_holdout_rolls_back_candidate():
    action, verified, reason = _verified_update_outcome(
        probe_action="accept",
        safety_complete=True,
        holdout_action="reject",
    )

    assert action == "rollback"
    assert not verified
    assert reason == "holdout_reject"


def test_accept_requires_probe_safety_and_holdout():
    action, verified, reason = _verified_update_outcome(
        probe_action="accept",
        safety_complete=True,
        holdout_action="accept",
    )

    assert action == "accept"
    assert verified
    assert reason is None
