"""Legacy CLI for general crystal generation.

Prefer ``python -m scripts.cli.generation.generate`` for new workflows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgdit.generation.general import *  # noqa: F401,F403
from cgdit.generation.general import cli


if __name__ == "__main__":
    cli()
