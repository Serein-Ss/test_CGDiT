"""Backward-compatible symmetry generation API.

New code should import from :mod:`cgdit.generation.symmetry`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgdit.generation.symmetry import *  # noqa: F401,F403
