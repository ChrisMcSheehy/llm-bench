"""The dashboard's inline scripts have to be valid JavaScript.

This exists because they were not, and nothing noticed for three days. Commit b56488f
put an HTML comment containing `speed` -- with backticks -- inside a template literal.
A backtick closes the literal, so the whole inline script failed to parse and the run
detail page rendered an empty shell. Every one of the 644 tests passed throughout,
because not one of them parsed or ran the page's JavaScript.

Two checks, deliberately. The first needs nothing installed and catches exactly the bug
that shipped. The second catches everything, and runs only where node exists - which is
most development machines and no guarantee, so it may not be the only guard.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

_STATIC = pathlib.Path(__file__).resolve().parent.parent / "llmbench" / "dashboard" / "static"
_PAGES = sorted(_STATIC.glob("*.html"))


def _inline_scripts(html: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", html, re.S)


@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.name)
def test_no_html_comment_contains_a_backtick(page):
    """A backtick in a comment inside a template literal terminates the literal.

    The comment reads as inert prose and is not: it is inside the string being built,
    so the parser sees it. Use quotes.
    """
    offenders = [c for c in re.findall(r"<!--.*?-->", page.read_text(encoding="utf-8"), re.S)
                 if "`" in c]
    assert not offenders, (
        f"{page.name} has an HTML comment containing a backtick, which closes any "
        f"template literal it sits inside:\n" + "\n".join(offenders))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.name)
def test_every_inline_script_parses(page, tmp_path):
    scripts = _inline_scripts(page.read_text(encoding="utf-8"))
    for i, script in enumerate(scripts):
        f = tmp_path / f"{page.stem}_{i}.js"
        f.write_text(script, encoding="utf-8")
        result = subprocess.run([shutil.which("node"), "--check", str(f)],
                                capture_output=True, text=True)
        assert result.returncode == 0, (
            f"{page.name} inline script {i} is not valid JavaScript:\n{result.stderr}")
