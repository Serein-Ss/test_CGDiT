"""Legacy CLI for the stability evaluation workflow.

Prefer ``python -m scripts.cli.evaluation.evaluate_stability``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import runpy

    runpy.run_module("cgdit.evaluation.stability", run_name="__main__")
else:
    from cgdit.evaluation.stability import *  # noqa: F401,F403
