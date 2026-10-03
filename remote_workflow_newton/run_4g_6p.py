#!/usr/bin/env python3
"""MP-20 workflow with validated one- or two-job training on one GPU."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SEEDS = (42, 123, 3407)
PROPS = {"fe": "formation_energy_per_atom", "bg": "band_gap"}
MODELS = ("mp20_base", "mp20_fe", "mp20_bg", "mp20_fe_bg")


def training_batch_overrides(train_batch, eval_batch):
    return [f"data.datamodule.batch_size.train={train_batch}",
            f"data.datamodule.batch_size.val={eval_batch}",
            f"data.datamodule.batch_size.test={eval_batch}",
            f"train.pl_trainer.accumulate_grad_batches={32 // train_batch}"]


def resume_history(root, previous, record):
    if previous['commit'] != record['commit']:
        changed = subprocess.check_output(
            ['git', '-c', 'core.quotepath=false', 'diff', '--name-only', previous['commit'], record['commit']],
            cwd=root, text=True).splitlines()
        if any(not (path.startswith('remote_workflow_newton/') or path in
                    ('README.md', 'tests/test_remote_workflow.py')) for path in changed):
            raise RuntimeError('Model, configuration or other source changed; use a new --run-id.')
    history = previous.get('resume_history', []).copy()
    keys = ('commit', 'train_batch_size', 'eval_batch_size')
    old = {key: previous.get(key, {'train_batch_size': 32, 'eval_batch_size': 16}.get(key))
           for key in keys}
    if any(old[key] != record[key] for key in keys):
        history.append(old)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "smoke", "train", "generate", "evaluate", "all"))
    parser.add_argument("--run-id", default="newton_4g_6p_20261003")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--train-batch-size", type=int, choices=(16, 32), default=32)
    parser.add_argument("--eval-batch-size", type=int, choices=(8, 16), default=16)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if not (root / "conf/model/experiments/exp_mp20_base.yaml").is_file():
        parser.error("Run from the latest newton repository root.")
    out = root / "output" / args.run_id
    os.environ.update(PROJECT_ROOT=str(root), HYDRA_JOBS=str(out),
                      WANDB_DIR=str(out / "wandb"), WABDB_DIR=str(out / "wandb"),
                      WANDB_MODE="offline", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                      OMP_NUM_THREADS="2", HYDRA_FULL_ERROR="1")
    if sys.platform == 'linux':
        os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
    if not args.dry_run:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            lock = (out / "workflow.lock").open("w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another workflow stage is active for this run-id.")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest = out / "workflow.json"
        record = {"commit": commit, "generator_seed": 42, "predictor_seeds": SEEDS,
                  "models": MODELS, "properties": PROPS, "python": sys.executable,
                  "train_batch_size": args.train_batch_size, "eval_batch_size": args.eval_batch_size,
                  "accumulate_grad_batches": 32 // args.train_batch_size,
                  "samples_per_group": 4096, "fe_target": -1.5, "bg_target": 2.0}
        if manifest.exists():
            record['resume_history'] = resume_history(root, json.loads(manifest.read_text()), record)
        manifest.write_text(json.dumps(record, indent=2) + "\n")

    def execute(name, command, done=None):
        if done and done.exists():
            print("[SKIP]", name, flush=True)
            return
        print("[RUN]", name, shlex.join(command), flush=True)
        if args.dry_run:
            return
        with (out / "logs" / f"{name}.log").open("a") as log:
            log.write("\nCOMMAND " + shlex.join(command) + "\n")
            log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if done:
            from validate_concurrency import validate_outputs
            validate_outputs(command)
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text("Training and best-checkpoint testing completed.\n")
        print("[OK]", name, flush=True)

    common = ["logging.wandb.mode=offline", "logging.wandb.log_model=false",
              "logging.wandb_watch.log=null", "logging.val_check_interval=5",
              "data.preprocess_workers=4", "train.pl_trainer.accelerator=gpu",
              "train.pl_trainer.devices=1", "train.pl_trainer.precision=32",
              *training_batch_overrides(args.train_batch_size, args.eval_batch_size),
              "train.model_checkpoints.save_last=false"]

    training_jobs = []

    def train(name, model, data, directory, seed, prop=None, smoke=False):
        command = [sys.executable, "-m", "cgdit.run", f"data={data}", f"model={model}",
                   f"expname={name}", f"hydra.run.dir={directory}",
                   f"train.random_seed={seed}", *common]
        if prop:
            command += [f"data.prop={prop}", "data.train_max_epochs=300"]
        else:
            command += ["data.train_max_epochs=1000"]
        if smoke:
            command += ["data.train_max_epochs=1", "logging.val_check_interval=1",
                        "+train.pl_trainer.limit_train_batches=2",
                        "+train.pl_trainer.limit_val_batches=2",
                        "+train.pl_trainer.limit_test_batches=2"]
        if stage == "train":
            if not (directory / ".complete").exists():
                training_jobs.append((name, command, directory / ".complete"))
        else:
            execute(name, command, directory / ".complete")

    def generate(model, mode, conditioned, pilot=False):
        conditions = []
        if conditioned:
            if model in ("mp20_fe", "mp20_fe_bg"):
                conditions += ["formation_energy_per_atom=-1.5"]
            if model in ("mp20_bg", "mp20_fe_bg"):
                conditions += ["band_gap=2.0"]
        tag = {"mp20_fe": "fe_m1p5", "mp20_bg": "bg_2", "mp20_fe_bg": "fe_m1p5_bg_2"}.get(model, "uncond") if conditioned else "uncond"
        label = f"{mode}_{tag}_n4096_seed42"
        directory = out / "generators" / model
        if pilot:
            label = "test_" + label
            # Smoke weights live outside the formal result tree.
            directory = out / "smoke" / model
        command = [sys.executable, "-m", "scripts.cli.generation.generate",
                   "--model_path", str(directory), "--batch_size", "1" if pilot else "32",
                   "--num_batches_to_samples", "1" if pilot else "128",
                   "--seed", "42", "--label", label,
                   "--guidance_scale", "2.0" if conditions else "0.0", "--max_atoms", "20"]
        if mode == "abinitio_empirical":
            command += ["--ab_initio", "--use_empirical_prior"]
        for condition in conditions:
            command += ["--condition", condition]
        # Common post-hoc targets also permit direct comparison to unconditional baselines.
        command += ["--evaluation_target", "formation_energy_per_atom=-1.5",
                    "--evaluation_target", "band_gap=2.0"]
        stage = "test" if pilot else "formal"
        method = "abinitio_empirical" if mode == "abinitio_empirical" else "template"
        conditioning = "unconditional" if pilot or not conditions else "conditional"
        payload = directory / "generated_structures" / stage / method / conditioning / f"eval_gen_{label}.pt"
        if not payload.exists():
            execute("generate_" + model + "_" + label, command)
        diagnostic = payload.with_suffix('.diagnostics.json')
        if not args.dry_run:
            from validate_concurrency import generation_diagnostics
            if not diagnostic.exists():
                diagnostic.write_text(json.dumps(generation_diagnostics(payload), indent=2) + "\n")
            if not json.loads(diagnostic.read_text())["all_finite"]:
                print("[WARNING] Non-finite generated geometry; not a valid generation pass:", diagnostic, flush=True)
        return directory, label

    def groups():
        for model in MODELS:
            yield model, "template", False
            if model != "mp20_base":
                yield model, "template", True
            yield model, "abinitio_empirical", model != "mp20_base"

    stages = ("preflight", "smoke", "train", "generate", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "preflight":
            execute("verify_inputs", [sys.executable, "-m", "scripts.cli.data.verify_inputs"])
            probe = """import torch, torch_scatter, torch_sparse, torch_geometric, pytorch_lightning
from torch_scatter import scatter
print('torch',torch.__version__,'CUDA',torch.version.cuda,'Lightning',pytorch_lightning.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable in this shell'
print('GPU',torch.cuda.get_device_name(0))
x=torch.randn(8,3,device='cuda',requires_grad=True)
y=scatter(x,torch.tensor([0,0,0,0,1,1,1,1],device='cuda'),dim=0)
y.square().sum().backward()
assert torch.isfinite(x.grad).all()
print('CUDA scatter backward PASS')
"""
            execute("gpu_probe", [sys.executable, "-c", probe])
            execute("core_tests", [sys.executable, "-m", "pytest", "-q",
                    "tests/test_diffusion_sampling.py", "tests/test_generation_conditioning.py",
                    "tests/test_property_predictor.py", "tests/test_generated_property_evaluation.py"])
            if not args.dry_run:
                with (out / "environment.txt").open("w") as stream:
                    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
        elif stage == "smoke":
            for model in MODELS:
                train("smoke_" + model, "experiments/exp_" + model, "mp_20", out / "smoke" / model, 42, smoke=True)
            for key, prop in PROPS.items():
                train("smoke_" + key, "property_predictors/m3gnet/regression", "mp_20_surrogate",
                      out / "smoke" / key, 42, prop, smoke=True)
            generate("mp20_fe_bg", "template", True, pilot=True)
        elif stage == "train":
            workers = args.parallel
            if args.parallel == 2 and not args.dry_run:
                import importlib.metadata
                report = json.loads((out / "concurrency_validation.json").read_text())
                gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
                if (report["commit"] != commit
                        or report["gpu"] != gpu or report["torch_version"] != importlib.metadata.version('torch')
                        or report['batch_size'] != args.train_batch_size
                        or report.get('eval_batch_size', 16) != args.eval_batch_size
                        or report.get('accumulate_grad_batches', 1) != 32 // args.train_batch_size):
                    raise RuntimeError("Run validate_concurrency.py on this server and source commit before parallel training.")
                if not report['serial']['passed']:
                    raise RuntimeError('Serial validation failed; inspect the resource report.')
                if not report.get('parallel2_safe_in_trial') or report.get('recommended_workers', 1) == 1:
                    workers = 1
                    print('[FALLBACK] Validation recommends one worker; training sequentially.', flush=True)
            for model in MODELS:
                train(model, "experiments/exp_" + model, "mp_20", out / "generators" / model, 42)
            for key, prop in PROPS.items():
                for seed in SEEDS:
                    train(f"predictor_{key}_seed{seed}", "property_predictors/m3gnet/regression", "mp_20_surrogate",
                          out / "predictors" / key / f"seed{seed}", seed, prop)
            if training_jobs:
                if args.dry_run:
                    for name, command, done in training_jobs:
                        print(f"[TRAIN workers={workers}]", name, shlex.join(command))
                else:
                    from validate_concurrency import run_training_jobs
                    (out / "parallel_jobs.json").write_text(json.dumps([
                        {"index": index, "name": job[0], "command": job[1]}
                        for index, job in enumerate(training_jobs)], indent=2))
                    status = run_training_jobs([job[1] for job in training_jobs], workers, out / "logs", 2.0, 4.0)
                    (out / "parallel_training.json").write_text(json.dumps(status, indent=2))
                    for index in status["completed_indices"]:
                        done = training_jobs[index][2]
                        done.write_text("Training and best-checkpoint testing completed.\n")
                    if not status["passed"]:
                        raise RuntimeError(status["failure"])
        elif stage == "generate":
            for model, mode, conditioned in groups():
                generate(model, mode, conditioned)
        elif stage == "evaluate":
            failures = []
            for model, mode, conditioned in groups():
                # generate() resumes an absent group but skips an existing payload.
                directory, label = generate(model, mode, conditioned)
                metric = directory / "evaluations/structural_metrics" / f"eval_metrics_gen_{label}.json"
                if not metric.exists():
                    try:
                        execute("metrics_" + model + "_" + label,
                                [sys.executable, "-m", "scripts.cli.evaluation.evaluate_metrics",
                                 "--root_path", str(directory), "--tasks", "gen", "--label", label,
                                 "--gt_file", str(root / "data/mp_20/test.csv"),
                                 "--calc_prop", "false", "--num_workers", "1", "--seed", "42"])
                    except subprocess.CalledProcessError:
                        failures.append(model + ":" + label)
            for seed in SEEDS:
                execute(f"properties_seed{seed}",
                        [sys.executable, "-m", "scripts.cli.evaluation.evaluate_generated_properties",
                         "--root_path", str(out / "generators"),
                         "--fe_run", str(out / "predictors/fe" / f"seed{seed}"),
                         "--bg_run", str(out / "predictors/bg" / f"seed{seed}"),
                         "--predictor_label", f"seed{seed}",
                         "--output_dir", str(out / "summaries" / f"seed{seed}"),
                         "--batch_size", "32", "--num_workers", "4", "--device", "cuda"])
            if not args.dry_run:
                import numpy as np
                rows = []
                for key in PROPS:
                    for seed in SEEDS:
                        run = out / "predictors" / key / f"seed{seed}"
                        p = np.load(run / "test_preds.npy").reshape(-1)
                        t = np.load(run / "test_targets.npy").reshape(-1)
                        if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
                            raise RuntimeError(f"Invalid predictor test outputs: {run}")
                        rows.append(dict(property=key, seed=seed, n=len(t),
                                         mae=float(np.abs(p-t).mean()), rmse=float(np.sqrt(((p-t)**2).mean()))))
                with (out / "summaries/predictor_test_metrics.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            if failures:
                raise RuntimeError("Structural metric failures (property evaluation still ran): " + ", ".join(failures))
    print("Workflow stage completed; outputs:", out, flush=True)


if __name__ == "__main__":
    main()
