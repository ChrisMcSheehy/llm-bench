"""Suite configuration.

A suite is a YAML file describing which targets to test and which evaluators to
run against them (with per-evaluator config overrides). See suites/default.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_suite(path: str, require_targets: bool = True) -> dict[str, Any]:
    """Read a suite file.

    `require_targets` is relaxed when the caller supplies its own targets — running a
    suite against servers named on the command line takes the targets from there, so the
    file is not required to define any.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data.setdefault("name", Path(path).stem)
    data.setdefault("targets", [])
    data.setdefault("evaluators", {})
    if require_targets and not data["targets"]:
        raise ValueError(f"Suite {path} defines no targets")
    return data
