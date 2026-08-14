"""Legacy CLI for generating symmetry query JSON files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cli.tools.generate_symmetry_queries import cli, generate_json_files


if __name__ == "__main__":
    cli()
