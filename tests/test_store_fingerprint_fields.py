"""What is hashed must also be recorded, or a fork cannot be explained afterwards."""
from __future__ import annotations

import json
import sqlite3

from llmbench.models import ModelFingerprint
from llmbench.store import Store


def test_the_work_split_settings_survive_a_round_trip(tmp_path):
    db = tmp_path / "results.db"
    fp = ModelFingerprint(
        engine="llama.cpp", base_url="http://localhost:8080", model_id="Qwen3-8B",
        n_gpu_layers="99", n_batch=4096, n_ubatch=512, n_parallel=4,
        launch_settings_observed=True,
    )

    store = Store(str(db))
    written = store.upsert_fingerprint(fp)
    store.close()

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM fingerprint WHERE hash=?", (written,)).fetchone()
    conn.close()

    assert row["n_gpu_layers"] == "99"
    assert row["n_batch"] == 4096
    assert row["n_ubatch"] == 512
    assert row["n_parallel"] == 4
    # Whether the settings were observed has to survive too, or a reader cannot tell
    # a genuine "no flags set" row from one whose server never reported any.
    assert row["launch_settings_observed"] == 1


def test_two_gpu_splits_are_two_rows(tmp_path):
    """The whole point, seen from the database side."""
    db = tmp_path / "two.db"
    common = dict(engine="llama.cpp", base_url="http://localhost:8080",
                  model_id="Qwen3-8B")

    store = Store(str(db))
    store.upsert_fingerprint(ModelFingerprint(**common, n_gpu_layers="40"))
    store.upsert_fingerprint(ModelFingerprint(**common, n_gpu_layers="99"))
    store.close()

    conn = sqlite3.connect(str(db))
    count = conn.execute("SELECT COUNT(*) FROM fingerprint").fetchone()[0]
    conn.close()
    assert count == 2, "the two layer splits were filed as one configuration"


def test_the_memory_estimate_and_its_derivation_survive_a_round_trip(tmp_path):
    db = tmp_path / "mem.db"
    fp = ModelFingerprint(
        engine="llama.cpp", base_url="http://localhost:8080", model_id="Qwen3-8B",
        kv_cache_bytes=2_483_027_968,
        kv_cache_derivation={"architecture": "gemma4", "block_count": 48},
    )
    store = Store(str(db))
    written = store.upsert_fingerprint(fp)
    store.close()

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM fingerprint WHERE hash=?", (written,)).fetchone()
    conn.close()

    assert row["kv_cache_bytes"] == 2_483_027_968
    assert json.loads(row["kv_cache_derivation_json"])["architecture"] == "gemma4"


def test_an_unknown_estimate_is_stored_as_null_not_zero(tmp_path):
    """A zero in this column would be indistinguishable from a free configuration."""
    db = tmp_path / "unknown.db"
    store = Store(str(db))
    written = store.upsert_fingerprint(ModelFingerprint(
        engine="llama.cpp", base_url="u", model_id="m"))
    store.close()

    conn = sqlite3.connect(str(db))
    value = conn.execute("SELECT kv_cache_bytes FROM fingerprint WHERE hash=?",
                         (written,)).fetchone()[0]
    conn.close()
    assert value is None
