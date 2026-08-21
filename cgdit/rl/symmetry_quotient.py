"""Independent degrees of freedom used by the CGDiT sampling measure."""

import torch


def orbit_representative_mask(anchor_index: torch.Tensor) -> torch.Tensor:
    """Return the unique representative node of every symmetry orbit.

    ``anchor_index[i]`` is the global node index of node ``i``'s representative.
    PyG increments this index during batching because its name contains ``index``.
    """
    if anchor_index.ndim != 1:
        raise ValueError("anchor_index must be one-dimensional")
    nodes = torch.arange(anchor_index.numel(), device=anchor_index.device)
    mask = nodes == anchor_index
    if anchor_index.numel() and not torch.equal(
        anchor_index[anchor_index], anchor_index
    ):
        raise ValueError("anchor_index must point to an idempotent representative")
    return mask


def representative_indices(anchor_index: torch.Tensor) -> torch.Tensor:
    """Return the global node indices of independent orbit representatives."""
    return torch.nonzero(
        orbit_representative_mask(anchor_index), as_tuple=False
    ).flatten()


def select_representatives(values: torch.Tensor, anchor_index: torch.Tensor) -> torch.Tensor:
    """Select one value per symmetry orbit from a full-cell tensor."""
    if values.shape[0] != anchor_index.numel():
        raise ValueError("values and anchor_index must have the same leading size")
    return values[representative_indices(anchor_index)]


def broadcast_from_representatives(
    representative_values: torch.Tensor, anchor_index: torch.Tensor
) -> torch.Tensor:
    """Broadcast a representative-only tensor to all atoms in each orbit."""
    representatives = representative_indices(anchor_index)
    if representative_values.shape[0] != representatives.numel():
        raise ValueError("expected one value per orbit representative")

    lookup = torch.full(
        (anchor_index.numel(),), -1, dtype=torch.long, device=anchor_index.device
    )
    lookup[representatives] = torch.arange(
        representatives.numel(), device=anchor_index.device
    )
    orbit_positions = lookup[anchor_index]
    if torch.any(orbit_positions < 0):
        raise ValueError("anchor_index contains a non-representative target")
    return representative_values[orbit_positions]


def lattice_active_mask(crystal_family, spacegroup: torch.Tensor) -> torch.Tensor:
    """Return the non-degenerate lattice coordinates for each space group."""
    return crystal_family.masks[spacegroup].bool()


def quotient_degrees_of_freedom(
    anchor_index: torch.Tensor, lattice_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Report discrete, coordinate and lattice dimensions per structure.

    The present CGDiT sampler assigns three torus coordinates to every orbit
    representative. Special-position tangent ranks require extra site-symmetry
    metadata and are intentionally not inferred from orbit coset operations.
    """
    representatives = representative_indices(anchor_index)
    return {
        "atom": torch.ones(representatives.numel(), device=anchor_index.device),
        "coord": torch.full(
            (representatives.numel(),), 3.0, device=anchor_index.device
        ),
        "lattice": lattice_mask.sum(dim=-1),
    }
