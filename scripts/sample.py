"""Legacy CLI for symmetry-conditioned CIF generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cli.generation.generate_symmetry import main, build_parser, cli


if __name__ == "__main__":
    cli()
