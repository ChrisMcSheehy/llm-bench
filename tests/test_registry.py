"""Discovery must find every test module, whatever was imported before it."""
from __future__ import annotations

import subprocess
import sys


def test_importing_one_evaluator_does_not_hide_the_others():
    """A partially-populated registry must not read as a complete one.

    `get()` and `available()` used to treat a non-empty registry as proof that
    discovery had already run. Any code importing a single evaluator module first
    — a plugin, a test, the dashboard — therefore left the tool convinced that
    module was the only test in existence, and a suite naming any other failed
    with "no evaluator named ...".

    This runs in a fresh interpreter because import caching makes the state it
    checks impossible to recreate inside an already-loaded process.
    """
    probe = (
        "import llmbench.evaluators.coding\n"          # the partial import
        "from llmbench.registry import available\n"
        "print(' '.join(available()))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe],
                            capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    found = result.stdout.split()
    assert "coding" in found
    assert "mcqa" in found, f"discovery was suppressed; only found: {found}"
