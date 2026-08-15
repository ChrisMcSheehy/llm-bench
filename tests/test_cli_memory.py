"""The memory question, answered with no server and no model loaded."""
from __future__ import annotations

from typer.testing import CliRunner

from llmbench.cli import app
from tests.test_gguf import _kv, _string, _u32, _write_gguf, STRING, U32

runner = CliRunner()


def _tiny_model(tmp_path):
    return _write_gguf(tmp_path / "tiny.gguf", [
        _kv("general.architecture", STRING, _string("llama")),
        _kv("llama.block_count", U32, _u32(4)),
        _kv("llama.attention.head_count", U32, _u32(8)),
        _kv("llama.attention.head_count_kv", U32, _u32(2)),
        _kv("llama.attention.key_length", U32, _u32(64)),
        _kv("llama.attention.value_length", U32, _u32(64)),
        _kv("llama.embedding_length", U32, _u32(512)),
    ])


def test_it_answers_with_no_server_running(tmp_path):
    result = runner.invoke(app, ["memory", "--model", _tiny_model(tmp_path),
                                 "--ctx", "1024"])
    assert result.exit_code == 0, result.output
    assert "MiB" in result.output or "GiB" in result.output


def test_compressing_the_cache_shows_a_smaller_figure(tmp_path):
    model = _tiny_model(tmp_path)
    f16 = runner.invoke(app, ["memory", "--model", model, "--ctx", "4096",
                              "--cache-type-k", "f16", "--cache-type-v", "f16"])
    q8 = runner.invoke(app, ["memory", "--model", model, "--ctx", "4096",
                             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
    assert f16.exit_code == 0 and q8.exit_code == 0
    assert f16.output != q8.output, "the cache type made no difference to the output"


def test_an_unreadable_model_says_unknown_and_fails_loudly(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a model")
    result = runner.invoke(app, ["memory", "--model", str(bad), "--ctx", "4096"])
    assert result.exit_code != 0, "an unreadable model must not exit 0"
    assert "unknown" in result.output.lower()
