"""The documentation describes the code that exists, checked rather than remembered.

This exists because the same defect shipped twice. Both times a plan's verification
asked "does the new thing appear in the README?", both times it did — in the new
section — and both times the architecture sketch further down still described an older
program. A check derived from the edit list is blind exactly where the edit list is
blind (LESSONS.md, verification-should-exceed-the-plans-minimum).

So this asserts a property of the whole file instead: every module and every command
appears. It fails when someone adds either and forgets the map, which is the failure
that actually happened.
"""
from __future__ import annotations

import pathlib
import re

from typer.main import get_command

from llmbench.cli import app

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_README = (_ROOT / "README.md").read_text(encoding="utf-8")


def _architecture_block() -> str:
    """The fenced sketch under the README's Architecture heading."""
    after = _README.split("## Architecture", 1)
    assert len(after) == 2, "the README has no Architecture section"
    return after[1].split("```")[1]


#: Modules the sketch need not name: support code with no behaviour of its own, listed
#: authoritatively in docs/ironclad/ARCHITECTURE.md instead. Keeping this explicit means
#: adding a module is a deliberate decision to document it or to exempt it.
_NOT_IN_SKETCH = {"config", "models", "resources"}


def test_every_module_appears_in_the_readme_architecture_sketch():
    block = _architecture_block()
    modules = {p.stem for p in (_ROOT / "llmbench").glob("*.py")} - {"__init__"}
    missing = sorted(m for m in modules - _NOT_IN_SKETCH if m not in block)
    assert not missing, (
        f"modules missing from the README architecture sketch: {missing}. "
        f"Add them there, or to _NOT_IN_SKETCH with a reason.")


def test_every_cli_command_appears_in_the_readme_architecture_sketch():
    block = _architecture_block()
    commands = sorted(get_command(app).commands)
    missing = [c for c in commands if not re.search(rf"\b{re.escape(c)}\b", block)]
    assert not missing, (
        f"CLI commands missing from the README architecture sketch: {missing}")


def test_every_module_appears_in_the_architecture_layer_table():
    """ARCHITECTURE.md is the authoritative list, so it has no exemptions."""
    table = (_ROOT / "docs" / "ironclad" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    modules = {p.stem for p in (_ROOT / "llmbench").glob("*.py")} - {"__init__"}
    missing = sorted(m for m in modules if f"{m}.py" not in table)
    assert not missing, f"modules missing from ARCHITECTURE.md's layer table: {missing}"
