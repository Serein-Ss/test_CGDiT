from types import SimpleNamespace

import pytest
import torch

from cgdit.pl_modules.diffusion import Diffusion


class FakeCrystalFamily:
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


def make_batch():
    return SimpleNamespace(
        num_graphs=1,
        num_nodes=1,
        num_atoms=torch.tensor([1]),
        atom_types=torch.tensor([1]),
        lengths=torch.tensor([[1.0, 1.0, 1.0]]),
        angles=torch.tensor([[90.0, 90.0, 90.0]]),
        frac_coords=torch.tensor([[0.25, 0.25, 0.25]]),
        spacegroup=torch.tensor([1]),
        anchor_index=torch.tensor([0]),
        ops=torch.eye(4).unsqueeze(0),
        ops_inv=torch.eye(3).unsqueeze(0),
        batch=torch.tensor([0]),
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


@pytest.mark.parametrize("diff_ratio", [0.0, -0.1, 1.1])
def test_invalid_diff_ratio_is_rejected(diff_ratio):
    with pytest.raises(ValueError, match="diff_ratio"):
        Diffusion.sample(FakeDiffusionModel(), make_batch(), diff_ratio=diff_ratio)
