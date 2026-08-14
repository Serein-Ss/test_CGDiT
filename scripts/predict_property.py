"""Legacy CLI for crystal property prediction.

Prefer ``python -m scripts.cli.evaluation.predict_property``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgdit.evaluation.property import *  # noqa: F401,F403
from cgdit.evaluation.property import cli


if __name__ == "__main__":
    cli()
