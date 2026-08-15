"""The host survives a round trip, and old databases gain the new shape."""
from __future__ import annotations

import json
import sqlite3

from llmbench.models import HostFingerprint
from llmbench.store import Store


def _host() -> HostFingerprint:
    return HostFingerprint(
        os="Linux", arch="x86_64", cpu_count=8, os_release="6.8.0",
        total_memory_bytes=33_454_276_608,
        devices=[{"id": "Vulkan0", "backend": "Vulkan", "name": "AMD Radeon RX 7900 XTX",
                  "total_mib": 24560, "free_mib": 23749}])


def test_a_host_round_trips(tmp_path):
    store = Store(str(tmp_path / "h.db"))
    written = store.upsert_host(_host())
    store.close()

    conn = sqlite3.connect(str(tmp_path / "h.db"))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM host WHERE hash=?", (written,)).fetchone()
    conn.close()

    assert row["os"] == "Linux"
    assert json.loads(row["devices_json"])[0]["name"] == "AMD Radeon RX 7900 XTX"
    assert row["label"]


def test_the_same_machine_twice_is_one_row(tmp_path):
    """Upsert, not insert: a machine is recorded once however many runs it does."""
    store = Store(str(tmp_path / "h.db"))
    store.upsert_host(_host())
    store.upsert_host(_host())
    n = store.conn.execute("SELECT COUNT(*) FROM host").fetchone()[0]
    store.close()
    assert n == 1


def test_two_machines_are_two_rows(tmp_path):
    """The case the phase exists for: one configuration, two machines."""
    store = Store(str(tmp_path / "h.db"))
    a = store.upsert_host(_host())
    b = store.upsert_host(HostFingerprint(
        os="Windows", arch="AMD64", cpu_count=16, total_memory_bytes=64 * 1024 ** 3,
        devices=[{"id": "CUDA0", "backend": "CUDA", "name": "NVIDIA GeForce RTX 4090",
                  "total_mib": 24564, "free_mib": 24000}]))
    n = store.conn.execute("SELECT COUNT(*) FROM host").fetchone()[0]
    store.close()
    assert a != b
    assert n == 2


def test_an_older_database_gains_the_host_column_on_run(tmp_path):
    """The store already migrates; this proves the new column reaches an old file."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE run (run_id TEXT PRIMARY KEY, fp_hash TEXT, suite TEXT,"
        " status TEXT, started_at TEXT, finished_at TEXT, error TEXT);"
        "INSERT INTO run (run_id) VALUES ('old-run');")
    conn.commit()
    conn.close()

    store = Store(str(db))
    columns = {r[1] for r in store.conn.execute("PRAGMA table_info(run)")}
    rows = store.conn.execute("SELECT run_id FROM run").fetchall()
    store.close()

    assert "host_hash" in columns, "migration did not add host_hash"
    assert [tuple(r) for r in rows] == [("old-run",)], "migration dropped existing rows"
