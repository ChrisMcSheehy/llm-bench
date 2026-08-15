"""The results database must not depend on the current working directory."""
from __future__ import annotations

from llmbench.store import default_db_path


def test_default_db_path_is_absolute_and_stable(tmp_path, monkeypatch):
    monkeypatch.delenv("LLMBENCH_DB", raising=False)
    monkeypatch.chdir(tmp_path)
    first = default_db_path()
    monkeypatch.chdir(tmp_path.parent)
    second = default_db_path()
    assert first == second, "database location changed with the working directory"
    assert first.is_absolute()


def test_env_var_overrides_the_default(tmp_path, monkeypatch):
    override = tmp_path / "custom.db"
    monkeypatch.setenv("LLMBENCH_DB", str(override))
    assert default_db_path() == override
