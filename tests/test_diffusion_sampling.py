from types import SimpleNamespace

import pytest
import torch

from cgdit.pl_modules.diffusion import Diffusion
from cgdit.rl.rollout import recompute_trajectory_log_probs


class FakeCrystalFamily:
    masks = torch.ones((231, 6))

    def de_so3(self, lattices):
        return lattices

    def m2v(self, lattices):
        return torch.zeros((lattices.shape[0], 6), device=lattices.device)

    def v2m(self, values):
        eye = torch.eye(3, device=values.device)
        return eye.unsqueeze(0).repeat(values.shape[0], 1, 1)

    def proj_k_to_spacegroup(self, values, spacegroup):
        return values


class FakeD3PM:
    def sample_stationary(self, shape):
        return torch.full(shape, 100, dtype=torch.long)

    def sample_and_compute_posterior_q(self, x_0, t, samples, **kwargs):
        return torch.zeros((samples.shape[0], 101), device=samples.device), None


class FakeDiffusionModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_atom_types = 100
        self.crystal_family = FakeCrystalFamily()
        self.d3pm = FakeD3PM()
        self.beta_scheduler = SimpleNamespace(
            timesteps=3,
            alphas=torch.tensor([1.0, 0.9, 0.9]),
            alphas_cumprod=torch.tensor([1.0, 0.9, 0.81]),
            betas=torch.tensor([0.0, 0.1, 0.1]),
            sigmas=torch.tensor([0.0, 0.1, 0.1]),
        )
        self.sigma_scheduler = SimpleNamespace(
            sigmas=torch.tensor([0.0, 0.1, 0.2]),
            sigmas_norm=torch.ones(3),
            sigma_begin=1.0,
        )
        self.q_sample_inputs = []

    def q_sample(self, x_start, t, diffusion):
        self.q_sample_inputs.append(x_start.clone())
        return torch.zeros_like(x_start)

    def _get_model_output(self, time_emb, atom_types, frac_coords, crys_fam,
                          num_atoms, batch_idx, batch_obj, guidance_scale):
        logits = torch.zeros((atom_types.shape[0], self.num_atom_types))
        logits[:, 0] = 1.0
        return torch.zeros_like(crys_fam), torch.zeros_like(frac_coords), logits

    def time_embedding(self, times):
        return torch.zeros((times.shape[0], 4), device=times.device)


def make_batch(anchor_index=None):
    anchor_index = torch.tensor([0]) if anchor_index is None else anchor_index
    num_nodes = anchor_index.numel()
    return SimpleNamespace(
        num_graphs=1,
        num_nodes=num_nodes,
        num_atoms=torch.tensor([num_nodes]),
        atom_types=torch.arange(1, num_nodes + 1),
        lengths=torch.tensor([[1.0, 1.0, 1.0]]),
        angles=torch.tensor([[90.0, 90.0, 90.0]]),
        frac_coords=torch.full((num_nodes, 3), 0.25),
        spacegroup=torch.tensor([1]),
        anchor_index=anchor_index,
        ops=torch.eye(4).repeat(num_nodes, 1, 1),
        ops_inv=torch.eye(3).repeat(num_nodes, 1, 1),
        batch=torch.zeros(num_nodes, dtype=torch.long),
    )


def test_partial_denoising_uses_zero_based_atom_types_and_returns_trajectory():
    model = FakeDiffusionModel()

    final, trajectory = Diffusion.sample(model, make_batch(), diff_ratio=0.67)

    assert torch.equal(model.q_sample_inputs[0], torch.tensor([0]))
    assert final["atom_types"].shape == (1,)
    assert trajectory["all_frac_coords"].shape[0] == 3


def test_fixed_atom_types_complete_without_q_sample_unpack_error():
    model = FakeDiffusionModel()

    final, _ = Diffusion.sample(
        model,
        make_batch(),
        fixed_atom_types=torch.tensor([13]),
    )

    assert torch.equal(final["atom_types"], torch.tensor([13]))
    assert all(torch.equal(values, torch.tensor([13])) for values in model.q_sample_inputs)


def test_atom_sampling_is_shared_by_every_symmetry_orbit():
    model = FakeDiffusionModel()
    batch = make_batch(torch.tensor([0, 0, 2, 2]))

    final, trajectory = Diffusion.sample(model, batch)

    assert final["atom_types"][0] == final["atom_types"][1]
    assert final["atom_types"][2] == final["atom_types"][3]
    assert all(
        torch.equal(step[0:1], step[1:2]) and torch.equal(step[2:3], step[3:4])
        for step in trajectory["all_atom_types"]
    )


def test_sample_rl_records_replayable_channel_actions_without_changing_sample_api():
    model = FakeDiffusionModel()
    batch = make_batch(torch.tensor([0, 0, 2, 2]))

    final, stack, rl_trajectory = Diffusion.sample_rl(model, batch)

    assert final["atom_types"].shape == (4,)
    assert stack["all_frac_coords"].shape[0] == 3
    assert [step.timestep for step in rl_trajectory.transitions] == [2, 1]
    assert set(rl_trajectory.stacked_old_log_probs()) == {"lattice", "coord", "atom"}
    assert rl_trajectory.transitions[0].actions_by_channel["atom"].shape == (2,)
    assert not rl_trajectory.transitions[-1].stochastic


def test_recorded_old_probability_matches_replay_under_unchanged_policy():
    model = FakeDiffusionModel()
    batch = make_batch(torch.tensor([0, 0, 2, 2]))
    _, _, trajectory = Diffusion.sample_rl(model, batch)

    replayed = recompute_trajectory_log_probs(model, batch, trajectory, step_lr=1e-5)
    recorded = trajectory.stacked_old_log_probs()

    for channel in recorded:
        assert torch.allclose(replayed[channel], recorded[channel], atol=1e-5)


def test_noise_seed_reproduces_the_complete_continuous_and_discrete_trajectory():
    model = FakeDiffusionModel()
    batch = make_batch(torch.tensor([0, 0, 2, 2]))

    first = Diffusion.sample_rl(model, batch, noise_seed=17)[1]
    second = Diffusion.sample_rl(model, batch, noise_seed=17)[1]

    assert torch.equal(first["all_atom_types"], second["all_atom_types"])
    assert torch.equal(first["all_frac_coords"], second["all_frac_coords"])
    assert torch.equal(first["all_lattices"], second["all_lattices"])


def test_probability_replay_can_subsample_long_trajectory_steps():
    model = FakeDiffusionModel()
    batch = make_batch(torch.tensor([0, 0, 2, 2]))
    _, _, trajectory = Diffusion.sample_rl(model, batch, noise_seed=23)

    replayed = recompute_trajectory_log_probs(
        model, batch, trajectory, step_lr=1e-5, transition_indices=[0]
    )
    recorded = trajectory.stacked_old_log_probs([0])

    assert all(values.shape == (1, 1) for values in replayed.values())
    for channel in recorded:
        assert torch.allclose(replayed[channel], recorded[channel], atol=1e-5)


def test_sample_rl_can_retain_only_replayed_steps_without_shortening_denoising():
    model = FakeDiffusionModel()
    batch = make_batch(torch.tensor([0, 0, 2, 2]))

    final, stack, trajectory = Diffusion.sample_rl(
        model,
        batch,
        noise_seed=23,
        replay_transitions=1,
        retain_trajectory_stack=False,
    )

    assert final["atom_types"].shape == (4,)
    assert stack is None
    assert [transition.timestep for transition in trajectory.transitions] == [2]
    assert model.beta_scheduler.timesteps == 3


@pytest.mark.parametrize("diff_ratio", [0.0, -0.1, 1.1])
def test_invalid_diff_ratio_is_rejected(diff_ratio):
    with pytest.raises(ValueError, match="diff_ratio"):
        Diffusion.sample(FakeDiffusionModel(), make_batch(), diff_ratio=diff_ratio)

class FakeForwardModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_atom_types = 100
        self.crystal_family = FakeCrystalFamily()
        self.beta_scheduler = SimpleNamespace(
            uniform_sample_t=lambda batch_size, device: torch.tensor([1, 4], device=device),
            alphas_cumprod=torch.full((5,), 0.5),
            betas=torch.full((5,), 0.1),
        )
        self.sigma_scheduler = SimpleNamespace(
            sigmas=torch.ones(5),
            sigmas_norm=torch.ones(5),
        )
        self.captured = {}

    def time_embedding(self, times):
        return torch.zeros((times.shape[0], 4), device=times.device)

    def conditioner(self, batch):
        return torch.zeros((batch.num_graphs, 4))

    def q_sample(self, x_start, t, diffusion, return_logits):
        self.captured["q_sample_t"] = t
        return x_start

    def decoder(self, time_emb, atom_types, frac_coords, crys_fam, num_atoms, batch):
        return (
            torch.zeros_like(crys_fam),
            torch.zeros_like(frac_coords),
            torch.zeros((frac_coords.shape[0], 101)),
        )

    def loss_fn(self, **kwargs):
        self.captured["loss_t"] = kwargs["t_discrete"]
        return {"loss": torch.tensor(0.0)}


def test_d3pm_forward_uses_q_t_and_zero_based_loss_t():
    model = FakeForwardModel()
    model.d3pm = object()
    batch = SimpleNamespace(
        num_graphs=2,
        num_atoms=torch.tensor([2, 1]),
        atom_types=torch.tensor([1, 2, 3]),
        lengths=torch.ones((2, 3)),
        angles=torch.full((2, 3), 90.0),
        frac_coords=torch.zeros((3, 3)),
        spacegroup=torch.tensor([1, 1]),
        anchor_index=torch.tensor([0, 0, 2]),
        ops=torch.eye(4).repeat(3, 1, 1),
        ops_inv=torch.eye(3).repeat(3, 1, 1),
        batch=torch.tensor([0, 0, 1]),
    )

    Diffusion.forward(model, batch)

    assert torch.equal(model.captured["q_sample_t"], torch.tensor([1, 4]))
    assert torch.equal(model.captured["loss_t"], torch.tensor([0, 0, 3]))
