"""Legacy CLI for reconstruction/CSP structure generation.

Despite the historical name, this command generates ``eval_diff_*.pt`` files.
Prefer ``python -m scripts.cli.generation.generate_reconstruction``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgdit.generation.reconstruction import *  # noqa: F401,F403
from cgdit.generation.reconstruction import cli


if __name__ == "__main__":
    cli()
