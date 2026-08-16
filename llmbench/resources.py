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

import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

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


#: A suite name the dashboard is allowed to ask for. Letters, digits, dash, underscore -
#: no dots and no separators, so no name can climb out of the directories below.
_SUITE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def suites_dir() -> Path:
    """Where a user's own suites live, beside the database and the profiles file."""
    return Path.home() / ".llmbench" / "suites"


def available_suites() -> dict[str, Path]:
    """Suite name -> file, for every suite that already exists on disk.

    Bundled ones first, then the user's own, so a personal `default.yaml` shadows the
    packaged one - which is the way round people expect.
    """
    found: dict[str, Path] = {}
    for directory in (data_path("suites"), suites_dir()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            if _SUITE_NAME.match(path.stem):
                found[path.stem] = path
    return found


def resolve_suite(name: str) -> Path:
    """The file for a suite *name*, or a refusal (design B6, restating L1).

    The dashboard may name a suite; it may never supply one. A name is checked against a
    pattern before it is used and then looked up in a listing of files that already exist,
    so the set of things the web layer can run stays exactly the set the user wrote to
    disk. A browser that could post a path - or a suite body, which names targets, and a
    target is an address and an argument list - would hand back everything decision L1
    withheld when it stopped the browser choosing which binary runs.
    """
    if not _SUITE_NAME.match(name or ""):
        raise ValueError(
            f"{name!r} is not a suite name. Names are letters, digits, dashes and "
            f"underscores - never a path.")
    suites = available_suites()
    if name not in suites:
        raise ValueError(
            f"no suite named {name!r}. Available: {sorted(suites) or 'none'}. "
            f"Put your own in {suites_dir()}.")
    return suites[name]


def load_jsonl(configured: Optional[str], *default_parts: str,
               limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Read a question file: one JSON object per line, blank lines ignored (design E5).

    Four evaluators carried their own copy of this, which meant four places for a
    malformed file to raise `Expecting value: line 1 column 1` naming neither the file
    nor which line was wrong - and a question file is exactly the thing a user edits by
    hand. The error below names both.

    Reading lives here beside `resolve_data_file` on purpose. This module is already the
    only place that knows where bundled data is, and splitting "which file" from "read
    that file" across two modules would make a new test module's author find both.
    """
    path = resolve_data_file(configured, *default_parts)
    items: list[dict[str, Any]] = []
    # enumerate from 1: line numbers in every editor and every other error message in
    # this project are 1-based, and an off-by-one here sends the reader to the wrong row.
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}: line {lineno} is not valid JSON ({exc.msg}). Each line must be "
                f"one complete JSON object; a question file is not a single JSON array."
            ) from exc
        if not isinstance(item, dict):
            # Valid JSON of the wrong shape. Left through, it reaches an evaluator as
            # `it["question"]` on a list or a string, and the error names a line of
            # framework code rather than the line of the file that is wrong.
            raise ValueError(
                f"{path}: line {lineno} is a JSON {type(item).__name__}, not an object. "
                f"Each line must be one question, e.g. "
                f'{{"id": "q1", "question": "...", "answer": "..."}}.')
        items.append(item)
    return items[:limit] if limit else items
