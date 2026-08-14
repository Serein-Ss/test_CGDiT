"""Stability, novelty, and distribution evaluation CLI."""

import runpy


if __name__ == "__main__":
    runpy.run_module("cgdit.evaluation.stability", run_name="__main__")
