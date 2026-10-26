"""Check calibration excludes the first token and preserves sign regimes."""

import torch

from sweep.neuron_stats import RegimeStats


def test_sign_regimes_ignore_first_token():
    stats = RegimeStats(1, "cpu")
    stats.gate = torch.tensor([[[100.0], [1.0], [-1.0], [1.0], [-1.0]]])
    stats.up = torch.tensor([[[100.0], [2.0], [-3.0], [-4.0], [5.0]]])
    activation = torch.nn.functional.silu(stats.gate) * stats.up
    stats.update(activation)
    assert torch.allclose(stats.pp_max, activation[0, 1])
    assert torch.allclose(stats.nn_max, activation[0, 2])
    assert torch.allclose(stats.pn_min, activation[0, 3])
    assert torch.allclose(stats.np_min, activation[0, 4])
    stats.update_resid(torch.tensor([[[999.0, 999.0], [3.0, 4.0]]]))
    assert stats.resid_norms[0].tolist() == [5.0]
