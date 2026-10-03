import sys

import numpy as np
import pytest

from remote_workflow_newton import validate_concurrency as validation
from remote_workflow_newton import run_4g_6p as workflow


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


def test_smaller_batch_preserves_effective_batch():
    values = dict(item.split('=', 1) for item in workflow.training_batch_overrides(16, 8))
    assert int(values['data.datamodule.batch_size.train']) * int(values['train.pl_trainer.accumulate_grad_batches']) == 32
    assert values['data.datamodule.batch_size.val'] == values['data.datamodule.batch_size.test'] == '8'


def test_workflow_only_update_records_resume_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.subprocess, 'check_output', lambda *args, **kwargs: 'remote_workflow_newton/run_4g_6p.py\n')
    old = {'commit': 'old'}
    new = {'commit': 'new', 'train_batch_size': 16, 'eval_batch_size': 8}
    assert workflow.resume_history(tmp_path, old, new) == [
        {'commit': 'old', 'train_batch_size': 32, 'eval_batch_size': 16}]


def test_model_update_prevents_checkpoint_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.subprocess, 'check_output', lambda *args, **kwargs: 'cgdit/pl_modules/diffusion.py\n')
    with pytest.raises(RuntimeError, match='source changed'):
        workflow.resume_history(tmp_path, {'commit': 'old'},
                                {'commit': 'new', 'train_batch_size': 16, 'eval_batch_size': 8})


def test_memory_stop_falls_back_without_repeating_completed_jobs(tmp_path, monkeypatch):
    calls = []
    def fake_jobs(commands, workers, *args, **kwargs):
        calls.append((commands, workers, kwargs))
        if workers == 2:
            return dict(passed=False, failure='Memory reserve breached: GPU', seconds=3,
                        completed_indices=[1])
        return dict(passed=True, failure=None, seconds=4, completed_indices=kwargs['job_indices'])
    monkeypatch.setattr(validation, 'run_jobs', fake_jobs)
    result = validation.run_training_jobs([['a'], ['b'], ['c']], 2, tmp_path, 2, 4)
    assert calls[1] == ([['a'], ['c']], 1, {'job_indices': [0, 2]})
    assert result['passed'] and result['completed_indices'] == [0, 1, 2]
    assert result['seconds'] == 7
    assert not result['parallel_attempt']['passed']


def test_nonmemory_training_failure_does_not_retry(tmp_path, monkeypatch):
    calls = []
    def fake_jobs(*args, **kwargs):
        calls.append(args)
        return dict(passed=False, failure='Training process exited 1', completed_indices=[])
    monkeypatch.setattr(validation, 'run_jobs', fake_jobs)
    assert not validation.run_training_jobs([['a']], 2, tmp_path, 2, 4)['passed']
    assert len(calls) == 1


def test_original_job_indices_survive_serial_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, 'gpu_memory', lambda: (24, 20))
    monkeypatch.setattr(validation, 'available_ram', lambda: 32)
    monkeypatch.setattr(validation, 'validate_outputs', lambda command: None)
    result = validation.run_jobs([[sys.executable, '-c', 'pass']], 1, tmp_path, 2, 4, job_indices=[7])
    assert result['passed'] and result['completed_indices'] == [7]
    assert (tmp_path/'job7.log').is_file()
