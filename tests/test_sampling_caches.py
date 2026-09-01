import torch

from cgdit.pl_modules.decoder.cspnet import CSPNet
from cgdit.pl_modules.diffusion import Diffusion

from tests.test_diffusion_sampling import FakeDiffusionModel, make_batch


class CountingConditioner:
    def __init__(self):
        self.calls = []

    def __call__(self, batch, force_mask=False):
        self.calls.append(force_mask)
        return torch.zeros((batch.num_graphs, 4))


class CountingDecoder:
    def __init__(self):
        self.calls = 0
        self.edge_cache_ids = []
        self.edge_builds = 0

    def gen_edges(self, num_atoms, frac_coords):
        self.edge_builds += 1
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return edge_index, torch.zeros((1, 3))

    def __call__(
        self,
        time_emb,
        atom_types,
        frac_coords,
        crys_fam,
        num_atoms,
        batch_idx,
        edge_cache=None,
    ):
        self.calls += 1
        self.edge_cache_ids.append(id(edge_cache))
        logits = torch.zeros((atom_types.shape[0], 100))
        logits[:, 0] = 1.0
        return torch.zeros_like(crys_fam), torch.zeros_like(frac_coords), logits


class CachedFakeDiffusionModel(FakeDiffusionModel):
    def __init__(self):
        super().__init__()
        self.training = False
        self.conditioner = CountingConditioner()
        self.decoder = CountingDecoder()

    def _get_model_output(self, *args, **kwargs):
        return Diffusion._get_model_output(self, *args, **kwargs)


def test_zero_guidance_reuses_sampling_caches_and_skips_conditional_branch():
    model = CachedFakeDiffusionModel()

    Diffusion.sample(model, make_batch(), guidance_scale=0.0)

    assert model.conditioner.calls == [True]
    assert model.decoder.edge_builds == 1
    assert model.decoder.calls == 4
    assert len(set(model.decoder.edge_cache_ids)) == 1


def test_cspnet_static_edge_cache_matches_uncached_forward():
    torch.manual_seed(7)
    model = CSPNet(
        hidden_dim=8,
        latent_dim=4,
        num_layers=1,
        num_freqs=2,
        pred_type=True,
        ln=False,
    ).eval()
    time_emb = torch.randn((1, 4))
    atom_types = torch.tensor([5, 7])
    frac_coords = torch.rand((2, 3))
    lattice = torch.randn((1, 6))
    num_atoms = torch.tensor([2])
    node2graph = torch.zeros(2, dtype=torch.long)

    expected = model(
        time_emb, atom_types, frac_coords, lattice, num_atoms, node2graph
    )
    edge_index, _ = model.gen_edges(num_atoms, frac_coords)
    edge_cache = (edge_index, node2graph[edge_index[0]])
    actual = model(
        time_emb,
        atom_types,
        frac_coords,
        lattice,
        num_atoms,
        node2graph,
        edge_cache=edge_cache,
    )

    for expected_value, actual_value in zip(expected, actual):
        assert torch.equal(expected_value, actual_value)
