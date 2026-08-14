"""Deprecated template generation CLI.

Use ``python -m scripts.cli.generation.generate`` for new runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.legacy.template_generation import *  # noqa: F401,F403
from scripts.legacy.template_generation import cli


if __name__ == "__main__":
    cli()
