"""Launch arguments that change the numbers must be readable from the argv."""
from __future__ import annotations

import pytest

from llmbench.targets.llamacpp import _parse_args


@pytest.mark.parametrize("argv, key, expected", [
    (["-b", "4096"], "n_batch", "4096"),
    (["--batch-size", "4096"], "n_batch", "4096"),
    (["-ub", "512"], "n_ubatch", "512"),
    (["--ubatch-size", "512"], "n_ubatch", "512"),
    (["-np", "4"], "n_parallel", "4"),
    (["--parallel", "4"], "n_parallel", "4"),
    (["-ngl", "99"], "ngl", "99"),
    (["--gpu-layers", "all"], "ngl", "all"),
])
def test_each_flag_and_alias_is_read(argv, key, expected):
    assert _parse_args(argv).get(key) == expected


def test_a_realistic_command_line_is_read_whole():
    argv = ["-m", "qwen.gguf", "-c", "16384", "-ngl", "99", "-b", "4096",
            "-ub", "512", "-np", "4", "-fa", "on"]
    parsed = _parse_args(argv)
    assert parsed["ngl"] == "99"
    assert parsed["n_batch"] == "4096"
    assert parsed["n_ubatch"] == "512"
    assert parsed["n_parallel"] == "4"
    assert parsed["flash_attn"] == "on"      # must not have been disturbed
    assert parsed["ctx"] == "16384"
