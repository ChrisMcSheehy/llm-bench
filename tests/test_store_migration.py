"""A database created by an older version must gain new columns when it is opened."""
from __future__ import annotations

import sqlite3

from llmbench.store import Store

# The fingerprint table exactly as it stood before this plan, frozen here on purpose.
# A migration test has to start from the real historical shape; building the "old"
# database from the current SCHEMA constant would test nothing, because the constant
# moves with the code.
_FINGERPRINT_TABLE_BEFORE_PHASE_2 = """
CREATE TABLE fingerprint (
  hash            TEXT PRIMARY KEY,
  engine          TEXT, engine_version TEXT,
  build_number    INTEGER, build_commit TEXT,
  base_url        TEXT, model_id TEXT, model_name TEXT,
  quant           TEXT, n_params TEXT, n_ctx INTEGER,
  kv_cache_k      TEXT, kv_cache_v TEXT, flash_attn TEXT,
  spec_type       TEXT, draft_model TEXT, mtp INTEGER,
  label           TEXT,
  sampling_json   TEXT, chat_template_sha TEXT,
  launch_args_json TEXT, raw_json TEXT,
  first_seen      TEXT, last_seen TEXT
);
"""

_NEW_COLUMNS = {"n_gpu_layers", "n_batch", "n_ubatch", "n_parallel",
                "launch_settings_observed", "kv_cache_bytes",
                "kv_cache_derivation_json"}


def test_an_older_database_gains_the_new_columns_when_opened(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_FINGERPRINT_TABLE_BEFORE_PHASE_2)
    conn.execute("INSERT INTO fingerprint (hash, model_id) VALUES ('oldhash', 'qwen')")
    conn.commit()
    conn.close()

    Store(str(db)).close()          # opening it is what must migrate it

    conn = sqlite3.connect(str(db))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(fingerprint)")}
    rows = conn.execute("SELECT hash FROM fingerprint").fetchall()
    conn.close()

    assert _NEW_COLUMNS <= columns, f"missing after migration: {_NEW_COLUMNS - columns}"
    assert rows == [("oldhash",)], "migration must not drop the existing rows"


def test_migrating_twice_is_harmless(tmp_path):
    """Every process that opens the database runs the migration, so it runs constantly."""
    db = tmp_path / "repeat.db"
    Store(str(db)).close()
    Store(str(db)).close()          # must not raise "duplicate column name"

    conn = sqlite3.connect(str(db))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(fingerprint)")}
    conn.close()
    assert _NEW_COLUMNS <= columns
