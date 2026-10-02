import sys

import numpy as np
import pytest

from remote_workflow_newton import validate_concurrency as validation


def test_two_workers_complete_all_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, 'gpu_memory', lambda: (24, 20))
    monkeypatch.setattr(validation, 'available_ram', lambda: 32)
    monkeypatch.setattr(validation, 'validate_outputs', lambda command: None)
    commands = [[sys.executable, '-c', 'pass'] for _ in range(4)]
    result = validation.run_jobs(commands, 2, tmp_path, 2, 4)
    assert result['passed']
    assert sorted(result['completed_indices']) == [0, 1, 2, 3]


def test_insufficient_memory_prevents_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, 'gpu_memory', lambda: (24, 1))
    monkeypatch.setattr(validation, 'available_ram', lambda: 32)
    result = validation.run_jobs([[sys.executable, '-c', 'pass']], 2, tmp_path, 2, 4)
    assert not result['passed']
    assert result['completed_indices'] == []
    assert not list(tmp_path.glob('job*.log'))


def test_failed_job_is_not_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, 'gpu_memory', lambda: (24, 20))
    monkeypatch.setattr(validation, 'available_ram', lambda: 32)
    monkeypatch.setattr(validation, 'validate_outputs', lambda command: None)
    result = validation.run_jobs([[sys.executable, '-c', 'raise SystemExit(1)']], 2, tmp_path, 2, 4)
    assert not result['passed']
    assert result['completed_indices'] == []


def test_invalid_predictor_outputs_fail_validation(tmp_path):
    (tmp_path/'hparams.yaml').write_text('model: example')
    (tmp_path/'epoch=0.ckpt').write_bytes(b'checkpoint fixture')
    np.save(tmp_path/'test_preds.npy', np.array([np.nan]))
    np.save(tmp_path/'test_targets.npy', np.array([1.0]))
    with pytest.raises(RuntimeError, match='Invalid best-checkpoint'):
        validation.validate_outputs([f'hydra.run.dir={tmp_path}', 'data.prop=band_gap'])


def test_nonfinite_generation_is_reported(tmp_path):
    import torch
    path = tmp_path/'generation.pt'
    torch.save({'frac_coords': torch.zeros(1, 3), 'lengths': torch.full((1, 3), float('inf')),
                'angles': torch.full((1, 3), float('nan')), 'num_atoms': torch.tensor([1])}, path)
    result = validation.generation_diagnostics(path)
    assert result['num_structures'] == 1
    assert result['finite_fields']['frac_coords']
    assert not result['all_finite']
