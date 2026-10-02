#!/usr/bin/env python3
"""Compare serial and two-worker GPU training; never starts formal training."""
import argparse
import json
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time


def available_ram():
    import psutil
    available = psutil.virtual_memory().available
    for base, limit_name, used_name in [(Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'),
                                       (Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes')]:
        try:
            limit = (base / limit_name).read_text().strip()
            if limit != 'max':
                available = min(available, max(0, int(limit) - int((base / used_name).read_text())))
        except (OSError, ValueError):
            pass
    return available / 1024**3


def gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--id=0',
                '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'], text=True)
    total, free = map(float, output.strip().split(','))
    return total / 1024, free / 1024


def validate_outputs(command):
    import numpy as np
    directory = Path(next(item.split('=', 1)[1] for item in command if item.startswith('hydra.run.dir=')))
    checkpoints = list(directory.glob('*.ckpt'))
    if not checkpoints or not all(path.stat().st_size for path in checkpoints) or not (directory/'hparams.yaml').is_file():
        raise RuntimeError(f'Missing trained checkpoint or config: {directory}')
    if any(item.startswith('data.prop=') for item in command):
        predictions = np.load(directory/'test_preds.npy')
        targets = np.load(directory/'test_targets.npy')
        if predictions.shape != targets.shape or not np.isfinite(predictions).all() or not np.isfinite(targets).all():
            raise RuntimeError(f'Invalid best-checkpoint test predictions: {directory}')


def generation_diagnostics(path):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    finite = {key: bool(torch.isfinite(payload[key]).all())
              for key in ('frac_coords', 'lengths', 'angles')}
    return dict(num_structures=int(payload['num_atoms'].numel()), finite_fields=finite,
                all_finite=all(finite.values()),
                note='Finite values do not establish physical or structural validity.')


def run_jobs(commands, workers, directory, gpu_reserve, ram_reserve):
    pending = list(enumerate(commands))
    active = []
    started = time.monotonic()
    min_gpu = min_ram = float('inf')
    failure = None
    completed = []
    try:
        while pending or active:
            _, free_gpu = gpu_memory()
            free_ram = available_ram()
            min_gpu, min_ram = min(min_gpu, free_gpu), min(min_ram, free_ram)
            if free_gpu < gpu_reserve or free_ram < ram_reserve:
                failure = 'Memory reserve breached; stopped only this validation batch.'
                break
            while pending and len(active) < workers:
                index, command = pending.pop(0)
                stream = (directory / f'job{index}.log').open('a')
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                active.append((index, process, stream))
            for index, process, stream in list(active):
                status = process.poll()
                if status is not None:
                    stream.close()
                    active.remove((index, process, stream))
                    if status:
                        failure = f'Training process exited {status}; inspect job logs.'
                        break
                    try:
                        validate_outputs(commands[index])
                    except (OSError, ValueError, RuntimeError) as error:
                        failure = str(error)
                        break
                    completed.append(index)
            if failure:
                break
            time.sleep(0.5)
    finally:
        for index, process, stream in active:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            stream.close()
    return dict(passed=failure is None, seconds=time.monotonic()-started,
                min_gpu_free_gib=min_gpu, min_ram_available_gib=min_ram, failure=failure,
                completed_indices=completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='newton_4g_6p_20261003')
    parser.add_argument('--gpu-reserve-gib', type=float, default=2.0)
    parser.add_argument('--ram-reserve-gib', type=float, default=4.0)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    out = root / 'output' / args.run_id
    trial = out / 'concurrency_validation' / time.strftime('%Y%m%d-%H%M%S')
    trial.mkdir(parents=True, exist_ok=False)
    os.environ.update(PROJECT_ROOT=str(root), HYDRA_JOBS=str(out), WANDB_DIR=str(out/'wandb'),
                      WANDB_MODE='offline', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2')
    # Serial preflight and smoke build shared full-data caches before concurrent readers.
    for stage in ('preflight', 'smoke'):
        subprocess.run([sys.executable, 'remote_workflow_newton/run_4g_6p.py', stage,
                        '--run-id', args.run_id], check=True)
    import pandas as pd
    from pymatgen.core import Structure
    # Include 20-site structures at every batch position to stress the MP-20 limit.
    stress = trial / 'data'
    stress.mkdir()
    source = pd.read_csv(root/'data/mp_20/train.csv')
    selected = []
    for index, row in source.iterrows():
        if len(Structure.from_str(row['cif'], fmt='cif')) == 20:
            selected.append(index)
            if len(selected) == 64:
                break
    if len(selected) < 64:
        raise RuntimeError('Could not find 64 twenty-site training examples.')
    frame = source.loc[selected].copy()
    for split in ('train', 'val', 'test'):
        frame.to_csv(stress/f'{split}.csv', index=False)
    # These repeated splits are for code/memory validation only, never scientific metrics.
    def commands(mode):
        jobs = [('joint_a', 'experiments/exp_mp20_fe_bg', 'mp_20', None),
                ('joint_b', 'experiments/exp_mp20_fe_bg', 'mp_20', None),
                ('fe', 'property_predictors/m3gnet/regression', 'mp_20_surrogate', 'formation_energy_per_atom'),
                ('bg', 'property_predictors/m3gnet/regression', 'mp_20_surrogate', 'band_gap')]
        result = []
        for name, model, data, prop in jobs:
            command = [sys.executable, '-m', 'cgdit.run', f'model={model}', f'data={data}',
                       f'data.root_path={stress}', f'hydra.run.dir={trial/mode/name}',
                       f'expname=memory_{name}', 'logging.wandb.mode=offline',
                       'logging.wandb.log_model=false', 'logging.wandb_watch.log=null',
                       'logging.val_check_interval=1', 'data.train_max_epochs=2',
                       'data.datamodule.batch_size.train=32', 'data.datamodule.batch_size.val=16',
                       'data.datamodule.batch_size.test=16', 'data.preprocess_workers=1',
                       'train.pl_trainer.accelerator=gpu', 'train.pl_trainer.devices=1',
                       'train.pl_trainer.precision=32', 'train.model_checkpoints.save_last=false']
            if prop:
                command.append(f'data.prop={prop}')
            result.append(command)
        return result
    # Warm stress caches once: avoid two writers targeting the same .pt file.
    import hydra
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(str(root/'conf'), version_base=None):
        cfg = compose(config_name='default', overrides=['data=mp_20', f'data.root_path={stress}', 'data.preprocess_workers=1'])
        module = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
        module.setup(); del module
    report = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=name','--format=csv,noheader'],text=True).strip(),
                  batch_size=32, precision=32, max_atoms=20, trial_directory=str(trial),
                  python=sys.executable, torch_version=importlib.metadata.version('torch'),
                  host_memory_note='Stress subsets understate full-dataset RAM; formal runs monitor container RAM and stop on reserve breach.')
    for mode, workers in [('serial',1), ('parallel2',2)]:
        directory = trial/mode
        directory.mkdir()
        report[mode] = run_jobs(commands(mode), workers, directory,
                                args.gpu_reserve_gib, args.ram_reserve_gib)
        if not report[mode]['passed']:
            break
    safe = report.get('parallel2',{}).get('passed',False)
    report['parallel2_safe_in_trial'] = safe
    report['speedup'] = report['serial']['seconds']/report['parallel2']['seconds'] if safe else None
    report['recommended_workers'] = 2 if safe and report['speedup'] > 1.05 else 1
    report['limits'] = 'Short wall-clock test includes imports/logging; not a guarantee of full-run memory or throughput.'
    (out/'concurrency_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not safe:
        raise SystemExit('Concurrent training not approved by this trial; use one worker.')


if __name__ == '__main__':
    main()
