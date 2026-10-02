"""Submit the frozen diagnostic campaign once, retaining each accepted job ID."""

import json
import os
from pathlib import Path
import subprocess

from scripts.basinguide_diagnostic import ROOT, checksum, write_json


def main():
    campaign = ROOT / "output/basinguide/bg3_500_seed42_20260914"
    manifest = campaign / "submission.json"
    if manifest.exists():
        raise RuntimeError("Submission manifest exists; inspect accepted jobs before any resubmission")
    assert (campaign / "preflight_passed.json").is_file()
    assert checksum(campaign / "models/checkpoint_sevennet_omni_i12.pth") == (
        "771543388f360d2762e09f62de685bfb85e9f637a06575f2545f03974d6f0522")
    records = dict(campaign=str(campaign), n_samples=500, target_eV=3.0, jobs={})
    env = dict(os.environ, CAMPAIGN=str(campaign))

    def submit(name, script, dependencies=None, action=None):
        command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes",
                   f"--output={campaign}/logs/{name}-%A_%a.out",
                   f"--error={campaign}/logs/{name}-%A_%a.err"]
        if dependencies:
            command.append(f"--dependency={dependencies}")
        if action:
            command.append(f"--export=ALL,ACTION={action}")
        command.append(str(ROOT / "submit_python" / script))
        response = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        job_id = response.stdout.strip().split(";")[0]
        assert job_id.isdigit(), response.stdout
        records["jobs"][name] = dict(job_id=job_id, dependency=dependencies, command=command)
        write_json(manifest, records)
        print(name, job_id, flush=True)
        return job_id

    generation = submit("generate", "basinguide_gpu.slurm", action="generate")
    mlip = submit("mlip", "basinguide_gpu.slurm", f"afterok:{generation}", "mlip")
    dft = submit("dft", "basinguide_dft.slurm", f"afterok:{generation}")
    submit("collect", "basinguide_collect.slurm", f"afterany:{mlip}:{dft}")


if __name__ == "__main__":
    main()
