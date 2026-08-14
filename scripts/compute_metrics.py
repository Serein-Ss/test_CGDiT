"""Legacy CLI for crystal generation and reconstruction metrics.

Prefer ``python -m scripts.cli.evaluation.evaluate_metrics``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgdit.evaluation.metrics import *  # noqa: F401,F403
from cgdit.evaluation.metrics import cli


if __name__ == "__main__":
    cli()
