"""Legacy CLI for the M3GNet property parity plot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cli.visualization.plot_property_results import cli


if __name__ == "__main__":
    cli()
