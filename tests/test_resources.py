"""Bundled data must be findable regardless of the current working directory."""
from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.resources import data_path, resolve_data_file


def test_bundled_dataset_is_found_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    found = data_path("datasets", "mcqa.jsonl")
    assert found.exists(), f"bundled dataset not found at {found}"
    assert found.read_text(encoding="utf-8").strip(), "bundled dataset is empty"


def test_bundled_coding_problems_are_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    problems = data_path("problems", "coding")
    assert problems.is_dir()
    names = sorted(p.name for p in problems.iterdir() if p.is_dir())
    assert names == ["balanced_brackets", "kadane", "rle_encode", "two_sum"]


def test_unset_data_file_falls_back_to_the_bundled_copy():
    got = resolve_data_file(None, "datasets", "mcqa.jsonl")
    assert got == data_path("datasets", "mcqa.jsonl")


def test_a_configured_but_missing_data_file_is_a_hard_error():
    with pytest.raises(FileNotFoundError) as exc:
        resolve_data_file("/no/such/file.jsonl", "datasets", "mcqa.jsonl")
    assert "/no/such/file.jsonl" in str(exc.value)
