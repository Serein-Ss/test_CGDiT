"""Backward-compatible imports for the refactored evaluation utilities.

New code should import from :mod:`cgdit.common.evaluation_utils` directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgdit.common.evaluation_utils import *  # noqa: F401,F403
