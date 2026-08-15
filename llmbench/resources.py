"""Locating the data files that ship with llmbench.

Everything bundled — question sets, coding problems, suite files — lives inside the
installed package under llmbench/data/. This module is the only place that knows that.

Why it exists: the defaults used to be bare relative paths like "datasets/mcqa.jsonl",
which only resolved when the tool was run from one particular folder. Run it anywhere
else and the evaluators quietly fell back to a handful of toy questions, then reported
those as a real result. A measurement tool that silently substitutes fake data is worse
than one that crashes, because the output still looks valid.
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Optional

# ironclad: importlib.resources over __file__ arithmetic — rejected Path(__file__).parent:
# the stdlib call is the supported way to reach package data and does not silently break
# if the package is ever laid out differently. It returns a real filesystem path for a
# normal pip install (pip always unpacks wheels), which is what lets the coding evaluator
# copy problem files around with ordinary path operations.
_DATA_ROOT = Path(str(files("llmbench"))) / "data"


def data_path(*parts: str) -> Path:
    """Absolute path to a bundled file, e.g. data_path("datasets", "mcqa.jsonl")."""
    return _DATA_ROOT.joinpath(*parts)


def resolve_data_file(configured: Optional[str], *default_parts: str) -> Path:
    """Pick the data file to use, and fail loudly if a configured one is missing.

    `configured` is whatever the suite file set (or None if it said nothing).
    `default_parts` locate the bundled fallback.

    A missing *configured* file raises: the user asked for a specific dataset and did
    not get it, and silently benchmarking something else instead would be a lie.
    """
    if configured:
        chosen = Path(configured)
        if not chosen.exists():
            raise FileNotFoundError(
                f"Configured data_file does not exist: {configured!r}. "
                f"Fix the path in your suite file, or remove the data_file line to use "
                f"the bundled default ({data_path(*default_parts)})."
            )
        return chosen
    return data_path(*default_parts)
