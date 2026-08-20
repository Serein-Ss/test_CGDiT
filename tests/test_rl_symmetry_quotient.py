import pytest
import torch

from cgdit.rl.symmetry_quotient import (
    broadcast_from_representatives,
    orbit_representative_mask,
    representative_indices,
    select_representatives,
)


def test_representative_selection_and_broadcast_support_batched_offsets():
    anchors = torch.tensor([0, 0, 2, 3, 3, 5])
    values = torch.tensor([10, 11, 20, 30, 31, 40])

    assert torch.equal(
        orbit_representative_mask(anchors),
        torch.tensor([True, False, True, True, False, True]),
    )
    assert torch.equal(representative_indices(anchors), torch.tensor([0, 2, 3, 5]))
    assert torch.equal(select_representatives(values, anchors), torch.tensor([10, 20, 30, 40]))
    assert torch.equal(
        broadcast_from_representatives(torch.tensor([1, 2, 3, 4]), anchors),
        torch.tensor([1, 1, 2, 3, 3, 4]),
    )


def test_invalid_anchor_mapping_is_rejected():
    with pytest.raises(ValueError, match="idempotent"):
        orbit_representative_mask(torch.tensor([1, 0]))
