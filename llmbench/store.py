"""SQLite results store.

Schema is deliberately relational and boring so you can point DuckDB, Tableau,
or plain SQL at data/llmbench.db and slice it however you like. JSON columns
(sampling, dims, raw) hold the flexible bits. One row per graded interaction in
`sample`, so the grain is easy to reason about.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from llmbench.models import (
    HostFingerprint, Metric, ModelFingerprint, RunResult, Sample,
)


def default_db_path() -> Path:
    """Where results live unless told otherwise.

    One fixed location under the user's home directory, so the bench and the dashboard
    always open the same database no matter which folder each was started from. It used
    to default to "data/llmbench.db" relative to the current directory, which meant
    running the bench from one folder and the dashboard from another opened two
    different databases and the dashboard appeared empty.

    The LLMBENCH_DB environment variable overrides it; the dashboard reads that same
    variable.
    """
    # ironclad: ~/.llmbench over platform-specific config dirs — rejected platformdirs
    # dep and hand-rolled XDG branching: one predictable path works everywhere and
    # nobody has asked for OS-native locations.
    override = os.environ.get("LLMBENCH_DB")
    if override:
        return Path(override)
    return Path.home() / ".llmbench" / "llmbench.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS fingerprint (
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
CREATE TABLE IF NOT EXISTS run (
  run_id       TEXT PRIMARY KEY,
  fp_hash      TEXT REFERENCES fingerprint(hash),
  suite        TEXT, status TEXT,
  started_at   TEXT, finished_at TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS sample (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id        TEXT REFERENCES run(run_id),
  evaluator     TEXT, case_id TEXT, grp TEXT,
  dims_json     TEXT,
  input_tokens  INTEGER, output_tokens INTEGER,
  latency_ms    REAL, tok_per_sec REAL,
  server_prompt_tps REAL, server_gen_tps REAL,
  score         REAL, passed INTEGER, error TEXT, meta_json TEXT,
  created_at    TEXT
);
CREATE TABLE IF NOT EXISTS metric (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id     TEXT REFERENCES run(run_id),
  evaluator  TEXT, name TEXT, value REAL, unit TEXT, dims_json TEXT,
  successes  INTEGER
);
CREATE TABLE IF NOT EXISTS hvote (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt_id  TEXT, mode TEXT,           -- pairwise | rating
  fp_a       TEXT, fp_b TEXT,
  winner     TEXT,                       -- a | b | tie | both_bad (pairwise)
  score      INTEGER,                    -- 1..5 (rating mode)
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS host (
  hash         TEXT PRIMARY KEY,
  os           TEXT, os_release TEXT, arch TEXT,
  cpu_count    INTEGER, cpu_model TEXT,
  total_memory_bytes INTEGER,
  devices_json TEXT, declared_json TEXT,
  label        TEXT,
  first_seen   TEXT, last_seen TEXT
);
CREATE INDEX IF NOT EXISTS ix_sample_run ON sample(run_id);
CREATE INDEX IF NOT EXISTS ix_metric_run ON metric(run_id);
CREATE INDEX IF NOT EXISTS ix_run_fp ON run(fp_hash);
"""


# Metrics that describe the model rather than the machine, and so may be compared and
# averaged across machines: the same model answers the same questions equally well
# wherever it runs. Anything NOT listed here is treated as machine-dependent and is
# grouped by host.
#
# This is deliberately an allowlist of what is safe to pool rather than a denylist of
# what is not. Getting it wrong in the pooling direction hides two machines inside one
# number, which is the failure Phase 4 exists to prevent; getting it wrong the other way
# costs a little statistical power and nothing else.
#
# `skipped_count` is deliberately absent: a skip caused by the time budget or by a
# machine running out of memory is a fact about that machine, so pooling it would
# average a laptop's four skips with a desktop's none. Left out, it groups by host.
QUALITY_METRICS = frozenset({
    "score_mean", "pass_rate", "pass@1", "problem_pass_rate", "accuracy",
    "prompt_acc", "instruction_acc", "recall", "effective_ctx", "perplexity",
    "build_score", "error_count", "responses",
    # How often the model answered at all is a fact about the model, not the machine: a
    # configuration that spends its budget thinking and returns nothing does so wherever
    # it runs (design B2). The speed metrics are deliberately absent from this set for
    # the opposite reason.
    "answer_rate",
})


def _sortable(value):
    """Order a view's axis labels sensibly whatever they are.

    Context lengths are numbers and must not sort as strings, where 1024 lands before 512.
    Task names are strings. A view may be declared over either, so numbers sort as numbers
    and everything else as text, with numbers first — a mixed axis is then at least stable
    rather than raising.
    """
    return (0, float(value), "") if isinstance(value, (int, float)) else (1, 0.0, str(value))


# Columns added after the first release. CREATE TABLE IF NOT EXISTS does nothing to a
# database that already exists, so anything added to SCHEMA below the initial release
# must also be listed here or it will never reach an existing file.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "fingerprint": {
        # Which executable produced the results. Two builds of one commit are two
        # programs, and without this they shared a fingerprint (design D6b).
        "binary_sha": "TEXT",
        "n_gpu_layers": "TEXT",
        "n_batch": "INTEGER",
        "n_ubatch": "INTEGER",
        "n_parallel": "INTEGER",
        "launch_settings_observed": "INTEGER",
        "kv_cache_bytes": "INTEGER",
        "kv_cache_derivation_json": "TEXT",
    },
    "run": {
        # Which machine the run happened on. Nullable: every run recorded before this
        # existed has no host, and that is unknown rather than any particular machine.
        "host_hash": "TEXT",
    },
    "sample": {
        # Why a rung was never attempted. NULL means it was; a reason means this row is
        # a gap in the data rather than a result of zero (design D3).
        "skipped": "TEXT",
        # Whether a gradable response arrived at all (design B2). 1 answered, 0 asked and
        # said nothing, NULL never successfully asked. Rows written before this existed
        # are NULL, which is correct: nobody recorded whether they answered, and a
        # backfilled guess would put a measurement where there was none.
        "answered": "INTEGER",
    },
    "metric": {
        # How many graded items this figure rests on. NULL means the aggregator did not
        # say, which is displayed as a dash and never as zero (design D7a).
        "n": "INTEGER",
        # The numerator behind a proportion (design C1). NULL means this figure is not a
        # proportion, or predates this column - both display as a dash and neither as a
        # zero. Deliberately not backfilled: the numerator is unrecoverable from a
        # rounded rate, and deriving it from the sample table would re-implement each
        # aggregator's filter, which is the drift D7a exists to prevent.
        "successes": "INTEGER",
    },
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any column listed in _ADDED_COLUMNS that this database does not have yet.

    Runs on every open, which is why it has to be idempotent: PRAGMA table_info is the
    check, and a column already present is skipped rather than re-added.

    The table and column names are interpolated into SQL because SQLite does not accept
    parameters in DDL. They are safe because they come from the constant above and never
    from anything a user supplies.
    """
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


class Store:
    def __init__(self, path: Optional[str] = None):
        db = Path(path) if path else default_db_path()
        db.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        _migrate(self.conn)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---- writes -----------------------------------------------------------
    def upsert_fingerprint(self, fp: ModelFingerprint) -> str:
        h = fp.fingerprint_hash
        now = fp.detected_at.isoformat()
        self.conn.execute(
            """INSERT INTO fingerprint
               (hash, engine, engine_version, build_number, build_commit, base_url,
                model_id, model_name, quant, n_params, n_ctx, kv_cache_k, kv_cache_v,
                flash_attn, spec_type, draft_model, mtp, label, sampling_json,
                chat_template_sha, launch_args_json, raw_json, first_seen, last_seen,
                n_gpu_layers, n_batch, n_ubatch, n_parallel,
                launch_settings_observed, kv_cache_bytes, kv_cache_derivation_json,
                binary_sha)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(hash) DO UPDATE SET last_seen=excluded.last_seen,
                 label=excluded.label, n_ctx=excluded.n_ctx""",
            (h, fp.engine, fp.engine_version, fp.build_number, fp.build_commit,
             fp.base_url, fp.model_id, fp.model_name, fp.quant, fp.n_params, fp.n_ctx,
             fp.kv_cache_k, fp.kv_cache_v, fp.flash_attn, fp.spec_type, fp.draft_model,
             int(fp.mtp), fp.label, json.dumps(fp.sampling), fp.chat_template_sha,
             json.dumps(fp.launch_args), json.dumps(fp.raw), now, now,
             fp.n_gpu_layers, fp.n_batch, fp.n_ubatch, fp.n_parallel,
             int(fp.launch_settings_observed), fp.kv_cache_bytes,
             json.dumps(fp.kv_cache_derivation), fp.binary_sha),
        )
        self.conn.commit()
        return h

    def upsert_host(self, host: HostFingerprint) -> str:
        h = host.host_hash
        now = host.detected_at.isoformat()
        self.conn.execute(
            """INSERT INTO host
               (hash, os, os_release, arch, cpu_count, cpu_model,
                total_memory_bytes, devices_json, declared_json, label,
                first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(hash) DO UPDATE SET last_seen=excluded.last_seen,
                 devices_json=excluded.devices_json""",
            (h, host.os, host.os_release, host.arch, host.cpu_count, host.cpu_model,
             host.total_memory_bytes, json.dumps(host.devices),
             json.dumps(host.declared), host.label, now, now),
        )
        self.conn.commit()
        return h

    def start_run(self, run: RunResult, host_hash: Optional[str] = None) -> None:
        """Record a run. `host_hash` is None when the machine could not be identified,
        which is also true of every run stored before hosts existed."""
        self.upsert_fingerprint(run.fingerprint)
        self.conn.execute(
            """INSERT INTO run (run_id, fp_hash, suite, status, started_at, host_hash)
               VALUES (?,?,?,?,?,?)""",
            (run.run_id, run.fingerprint.fingerprint_hash, run.suite, run.status,
             run.started_at.isoformat(), host_hash),
        )
        self.conn.commit()

    def add_samples(self, run_id: str, samples: list[Sample]) -> None:
        self.conn.executemany(
            """INSERT INTO sample
               (run_id, evaluator, case_id, grp, dims_json, input_tokens, output_tokens,
                latency_ms, tok_per_sec, server_prompt_tps, server_gen_tps, score,
                passed, error, skipped, answered, meta_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(run_id, s.evaluator, s.case_id, s.group, json.dumps(s.dims),
              s.input_tokens, s.output_tokens, s.latency_ms, s.tok_per_sec,
              s.server_prompt_tps, s.server_gen_tps, s.score,
              None if s.passed is None else int(s.passed), s.error, s.skipped,
              None if s.answered is None else int(s.answered),
              json.dumps(s.meta), s.created_at.isoformat()) for s in samples],
        )
        self.conn.commit()

    def add_metrics(self, run_id: str, metrics: list[Metric]) -> None:
        self.conn.executemany(
            """INSERT INTO metric
                 (run_id, evaluator, name, value, unit, n, successes, dims_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            [(run_id, m.evaluator, m.name, m.value, m.unit, m.n, m.successes,
              json.dumps(m.dims))
             for m in metrics],
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str, finished_at: str,
                   error: Optional[str] = None) -> None:
        self.conn.execute(
            "UPDATE run SET status=?, finished_at=?, error=? WHERE run_id=?",
            (status, finished_at, error, run_id))
        self.conn.commit()

    # ---- reads (dashboard) ------------------------------------------------
    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def runs(self) -> list[dict]:
        return self._rows(
            """SELECT r.run_id, r.suite, r.status, r.started_at, r.finished_at,
                      f.label, f.engine, f.model_name, f.quant, f.n_ctx,
                      f.kv_cache_k, f.kv_cache_v, f.spec_type, f.build_commit,
                      f.kv_cache_bytes,
                      r.host_hash, h.label AS host_label
               FROM run r JOIN fingerprint f ON f.hash = r.fp_hash
               LEFT JOIN host h ON h.hash = r.host_hash
               ORDER BY r.started_at DESC""")

    def metrics_for(self, run_id: str) -> list[dict]:
        return self._rows("SELECT evaluator, name, value, unit, n, successes, dims_json "
                          "FROM metric WHERE run_id=?", (run_id,))

    def leaderboard(self) -> list[dict]:
        rows = self._rows(
            """SELECT r.run_id, r.started_at, f.label, f.engine,
                      f.kv_cache_k, f.kv_cache_v, f.spec_type,
                      r.host_hash, h.label AS host_label,
                      m.evaluator, m.name, m.value, m.n, m.successes
               FROM run r JOIN fingerprint f ON f.hash=r.fp_hash
               LEFT JOIN host h ON h.hash=r.host_hash
               JOIN metric m ON m.run_id=r.run_id
               WHERE m.name IN ('score_mean','pass_rate','pass@1','decode_tps',
                                'prefill_tps','effective_ctx','error_count','perplexity',
                                'answer_rate')
                 AND json_extract(m.dims_json,'$')='{}'
               ORDER BY r.started_at DESC""")
        board: dict[str, dict] = {}
        for row in rows:
            b = board.setdefault(row["run_id"], {
                "run_id": row["run_id"], "label": row["label"], "engine": row["engine"],
                "started_at": row["started_at"],
                "kv": row["kv_cache_k"] if row["kv_cache_k"] == row["kv_cache_v"]
                     else f"{row['kv_cache_k']}/{row['kv_cache_v']}",
                "spec_type": row["spec_type"],
                # Which machine produced these figures. Without it two rows for one
                # configuration read as run-to-run variance when they are two machines.
                "host_hash": row["host_hash"],
                "host_label": row["host_label"] or "unknown machine",
            })
            # Both under predictable keys, so a caller cannot show one without the other
            # having been available to it (design D7b).
            b[f"{row['evaluator']}.{row['name']}"] = row["value"]
            b[f"{row['evaluator']}.{row['name']}.n"] = row["n"]
            # Third under the same predictable key, so a caller cannot show the figure
            # without its interval having been available to it (design D7b, C8).
            b[f"{row['evaluator']}.{row['name']}.successes"] = row["successes"]
        return list(board.values())

    def pooled_quality(self) -> list[dict]:
        """Quality figures averaged per configuration, across every machine.

        Only metrics in QUALITY_METRICS appear. A metric nobody has classified is left
        out rather than pooled, because pooling the wrong thing hides two machines
        inside one number (design D1).

        A proportion is pooled by summing successes and trials, never by averaging the
        per-run rates. Averaging weights a ten-item run as heavily as a hundred-item one:
        runs of 80/100 and 4/10 pool to 84/110 = 0.764, where the average of the rates is
        0.60 (design C3). A mean of means is also a quantity with no sample size, which
        is the other reason no interval could be put on the figure this used to produce.

        The average is kept only where a group is not wholly numbered — runs stored
        before `successes` existed carry NULL, and summing across a NULL would drop that
        run from the denominator without saying so, which is worse than the average it
        would replace.
        """
        rows = self._rows(
            """SELECT r.fp_hash, f.label, m.evaluator, m.name,
                      AVG(m.value) AS rate_mean, COUNT(*) AS runs,
                      CASE WHEN COUNT(m.n)=COUNT(*) THEN SUM(m.n) END AS items,
                      CASE WHEN COUNT(m.successes)=COUNT(*)
                           THEN SUM(m.successes) END AS successes
               FROM metric m JOIN run r ON r.run_id=m.run_id
               JOIN fingerprint f ON f.hash=r.fp_hash
               WHERE json_extract(m.dims_json,'$')='{}'
               GROUP BY r.fp_hash, m.evaluator, m.name
               ORDER BY f.label, m.evaluator, m.name""")
        out = []
        for row in rows:
            if row["name"] not in QUALITY_METRICS:
                continue
            # COUNT(col) skips NULLs, so `successes` is non-NULL here only when every run
            # in the group carried one — which is exactly when summing is honest.
            row["value"] = (row["successes"] / row["items"]
                            if row["successes"] is not None and row["items"]
                            else row["rate_mean"])
            del row["rate_mean"]
            out.append(row)
        return out

    def pooled_speed(self) -> list[dict]:
        """Everything else, grouped by configuration **and machine**.

        A run whose machine is unknown groups on its own: `host_hash IS NULL` is not a
        machine, and pooling it with a known one is the same mistake D6a refused to make
        for unobserved launch settings.
        """
        rows = self._rows(
            """SELECT r.fp_hash, f.label, r.host_hash, h.label AS host_label,
                      m.evaluator, m.name, AVG(m.value) AS value, COUNT(*) AS runs,
                      CASE WHEN COUNT(m.n)=COUNT(*) THEN SUM(m.n) END AS items
               FROM metric m JOIN run r ON r.run_id=m.run_id
               JOIN fingerprint f ON f.hash=r.fp_hash
               LEFT JOIN host h ON h.hash=r.host_hash
               WHERE json_extract(m.dims_json,'$')='{}'
               GROUP BY r.fp_hash, r.host_hash, m.evaluator, m.name
               ORDER BY f.label, h.label, m.evaluator, m.name""")
        return [r for r in rows if r["name"] not in QUALITY_METRICS]

    def configuration_effort(self) -> list[dict]:
        """How much testing each configuration has actually received.

        Derived from what is already stored rather than counted as it happens, so it
        cannot be inflated and cannot go stale (design D9a).

        Two queries merged in Python, deliberately not one. Joining `run` to `sample`
        and counting both in a single statement multiplies each run's row by that run's
        sample count, so a run with sixty samples would be counted sixty times.

        Failed and partial runs are reported alongside the total rather than removed
        from it: "six runs, two of which errored" is information, and showing only the
        four is not.
        """
        rows = self._rows(
            """SELECT r.fp_hash, f.label,
                      COUNT(*) AS runs,
                      SUM(CASE WHEN r.status='error'   THEN 1 ELSE 0 END) AS failed_runs,
                      SUM(CASE WHEN r.status='partial' THEN 1 ELSE 0 END) AS partial_runs,
                      MIN(r.started_at) AS first_run,
                      MAX(r.started_at) AS last_run,
                      COUNT(DISTINCT r.host_hash) AS machines
               FROM run r JOIN fingerprint f ON f.hash=r.fp_hash
               GROUP BY r.fp_hash
               ORDER BY f.label""")
        # COUNT(DISTINCT ...) ignores NULL, so a run whose machine was never identified
        # counts toward `runs` and not toward `machines`. That is the intended reading:
        # unknown is not a machine (design D9b).
        graded = {r["fp_hash"]: r["graded_samples"] for r in self._rows(
            """SELECT r.fp_hash, COUNT(*) AS graded_samples
               FROM sample s JOIN run r ON r.run_id=s.run_id
               WHERE s.error IS NULL AND s.skipped IS NULL
               GROUP BY r.fp_hash""")}
        for row in rows:
            row["graded_samples"] = graded.get(row["fp_hash"], 0)
        return rows

    def hosts(self) -> list[dict]:
        """Every machine seen, newest first."""
        return self._rows(
            """SELECT h.*, COUNT(r.run_id) AS runs
               FROM host h LEFT JOIN run r ON r.host_hash=h.hash
               GROUP BY h.hash ORDER BY h.last_seen DESC""")

    def view_data(self, run_id: str, evaluator: str, x: str,
                  y: Optional[str] = None, value: str = "score") -> dict:
        """Average `value` over one evaluator's samples, grouped by one or two dimensions.

        The generic replacement for the hand-written per-module queries that used to live
        here — a needle heatmap, a coding bar chart, a throughput line — each of which had
        an evaluator's name written into its SQL, so a new test module could not have a
        chart without this file being edited (design E3).

        Every cell carries its own count, because a colour drawn from one attempt reads
        exactly like the same colour drawn from five (design D7b). A cell nobody probed
        reports a count of zero and a value of None: no measurement is not a score of
        nought, and the renderer draws it as a gap.
        """
        if value not in ("score", "passed"):
            # These reach SQL. A module declaration is code rather than user input, but a
            # column name assembled from a string is not a habit worth having.
            raise ValueError(f"a view may average score or passed, not {value!r}")

        rows = self._rows(
            f"""SELECT dims_json, {value} AS v FROM sample
                WHERE run_id=? AND evaluator=?
                  AND error IS NULL AND skipped IS NULL AND {value} IS NOT NULL""",
            (run_id, evaluator))

        cells: dict[tuple, list[float]] = {}
        for row in rows:
            dims = json.loads(row["dims_json"])
            if x not in dims or (y is not None and y not in dims):
                continue          # a sample that carries no such label belongs in no cell
            key = (dims[x],) if y is None else (dims[x], dims[y])
            cells.setdefault(key, []).append(float(row["v"]))

        def mean(values: Optional[list[float]]) -> Optional[float]:
            return round(sum(values) / len(values), 4) if values else None

        xs = sorted({k[0] for k in cells}, key=_sortable)
        if y is None:
            return {"x": xs, "v": [mean(cells.get((k,))) for k in xs],
                    "n": [len(cells.get((k,), [])) for k in xs]}
        ys = sorted({k[1] for k in cells}, key=_sortable)
        return {"x": xs, "y": ys,
                "z": [[mean(cells.get((xv, yv))) for xv in xs] for yv in ys],
                "n": [[len(cells.get((xv, yv), [])) for xv in xs] for yv in ys]}

    def skipped(self, run_id: str) -> list[dict]:
        """Everything a run did not attempt, and why.

        A gap that is never explained reads as a figure of zero, so the reason travels
        with the gap (design D3).
        """
        return self._rows(
            """SELECT evaluator, case_id, grp, dims_json, skipped
               FROM sample WHERE run_id=? AND skipped IS NOT NULL
               ORDER BY evaluator, case_id""", (run_id,))

    def capabilities(self, run_id: str) -> list[dict]:
        """One headline 0..1 score per evaluator, for a capability bar/radar.

        Prefers score_mean, then pass_rate, then pass@1, then instruction_acc. The count
        travels with the score: this chart is where a figure is most often read on its
        own, and a bar drawn over six questions looks identical to one drawn over six
        hundred (design D7b).
        """
        rows = self._rows(
            """SELECT evaluator, name, value, n FROM metric
               WHERE run_id=? AND json_extract(dims_json,'$')='{}'
                 AND name IN ('score_mean','pass_rate','pass@1','instruction_acc')""",
            (run_id,))
        pref = {"score_mean": 0, "pass_rate": 1, "pass@1": 2, "instruction_acc": 3}
        best: dict[str, tuple[int, float, Optional[int]]] = {}
        for r in rows:
            rank = pref.get(r["name"], 9)
            if r["evaluator"] not in best or rank < best[r["evaluator"]][0]:
                best[r["evaluator"]] = (rank, r["value"], r["n"])
        return [{"evaluator": e, "score": v, "n": n}
                for e, (_, v, n) in sorted(best.items())]

    def perplexity(self, run_id: str):
        rows = self._rows(
            "SELECT value FROM metric WHERE run_id=? AND name='perplexity'", (run_id,))
        return rows[0]["value"] if rows else None

    def trend(self, evaluator: str, metric: str) -> list[dict]:
        return self._rows(
            """SELECT r.started_at, f.label, f.build_commit, f.build_number, m.value
               FROM metric m JOIN run r ON r.run_id=m.run_id
               JOIN fingerprint f ON f.hash=r.fp_hash
               WHERE m.evaluator=? AND m.name=? AND json_extract(m.dims_json,'$')='{}'
               ORDER BY r.started_at""", (evaluator, metric))

    # ---- human eval / arena ----------------------------------------------
    def gradable_responses(self) -> list[dict]:
        """Latest human-gradable output per (fingerprint, prompt), across the
        `human` (text) and `oneshot` (html artifact) evaluators. Carries a
        `render` hint and timing so the UI can show badges."""
        rows = self._rows(
            """SELECT s.id AS sid, s.evaluator, r.fp_hash, f.label,
                      json_extract(s.meta_json,'$.prompt_id')  AS prompt_id,
                      json_extract(s.meta_json,'$.kind')       AS kind,
                      json_extract(s.meta_json,'$.category')   AS category,
                      json_extract(s.meta_json,'$.prompt')     AS prompt,
                      json_extract(s.meta_json,'$.response')   AS response,
                      json_extract(s.meta_json,'$.artifact')   AS artifact,
                      json_extract(s.meta_json,'$.build_score') AS build_score,
                      s.latency_ms, s.tok_per_sec, s.output_tokens,
                      r.started_at
               FROM sample s JOIN run r ON r.run_id=s.run_id
               JOIN fingerprint f ON f.hash=r.fp_hash
               WHERE s.evaluator IN ('human','oneshot')
                 AND s.error IS NULL AND s.skipped IS NULL
               ORDER BY r.started_at""")
        latest: dict[tuple, dict] = {}
        for row in rows:
            row["render"] = "html" if row["evaluator"] == "oneshot" else "text"
            row["content"] = row["artifact"] if row["render"] == "html" else row["response"]
            latest[(row["fp_hash"], row["prompt_id"])] = row
        return list(latest.values())

    def human_responses(self) -> list[dict]:
        return [r for r in self.gradable_responses() if r["evaluator"] == "human"]

    def gallery(self) -> list[dict]:
        """One-shot artifacts grouped by prompt, with per-config timing — for the
        labelled gallery view."""
        rows = [r for r in self.gradable_responses() if r["evaluator"] == "oneshot"]
        by_prompt: dict[str, dict] = {}
        for r in rows:
            g = by_prompt.setdefault(r["prompt_id"], {
                "prompt_id": r["prompt_id"], "kind": r["kind"],
                "prompt": r["prompt"], "entries": []})
            g["entries"].append({
                "fp": r["fp_hash"], "label": r["label"], "html": r["content"],
                "latency_ms": r["latency_ms"], "tok_per_sec": r["tok_per_sec"],
                "output_tokens": r["output_tokens"],
                "build_score": r["build_score"]})
        return list(by_prompt.values())

    def add_vote(self, prompt_id, mode, fp_a=None, fp_b=None, winner=None, score=None):
        self.conn.execute(
            """INSERT INTO hvote (prompt_id, mode, fp_a, fp_b, winner, score, created_at)
               VALUES (?,?,?,?,?,?,datetime('now'))""",
            (prompt_id, mode, fp_a, fp_b, winner, score))
        self.conn.commit()

    def _labels(self) -> dict[str, str]:
        return {r["hash"]: r["label"]
                for r in self._rows("SELECT hash, label FROM fingerprint")}

    def arena_leaderboard(self) -> list[dict]:
        labels = self._labels()
        votes = self._rows(
            "SELECT fp_a, fp_b, winner FROM hvote WHERE mode='pairwise' ORDER BY id")
        elo: dict[str, float] = {}
        games: dict[str, int] = {}
        wins: dict[str, float] = {}
        K = 32.0
        for v in votes:
            a, b = v["fp_a"], v["fp_b"]
            for fp in (a, b):
                elo.setdefault(fp, 1000.0); games.setdefault(fp, 0); wins.setdefault(fp, 0.0)
            ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
            eb = 1.0 - ea
            if v["winner"] == "a":
                sa, sb = 1.0, 0.0
            elif v["winner"] == "b":
                sa, sb = 0.0, 1.0
            else:                              # tie | both_bad -> draw
                sa, sb = 0.5, 0.5
            elo[a] += K * (sa - ea); elo[b] += K * (sb - eb)
            games[a] += 1; games[b] += 1; wins[a] += sa; wins[b] += sb

        ratings = self._rows(
            """SELECT fp_a AS fp, AVG(score) AS avg_score, COUNT(*) AS n
               FROM hvote WHERE mode='rating' GROUP BY fp_a""")
        rmap = {r["fp"]: r for r in ratings}

        board = []
        for fp in set(list(elo) + list(rmap)):
            r = rmap.get(fp, {})
            board.append({
                "fp": fp, "label": labels.get(fp, fp[:8]),
                "elo": round(elo.get(fp, 1000.0), 1),
                "games": games.get(fp, 0),
                "win_rate": round(wins.get(fp, 0.0) / games[fp], 3) if games.get(fp) else None,
                "avg_stars": round(r["avg_score"], 2) if r.get("avg_score") is not None else None,
                "n_ratings": r.get("n", 0),
            })
        board.sort(key=lambda x: x["elo"], reverse=True)
        return board

    def vote_count(self) -> dict:
        pw = self._rows("SELECT COUNT(*) AS n FROM hvote WHERE mode='pairwise'")[0]["n"]
        rt = self._rows("SELECT COUNT(*) AS n FROM hvote WHERE mode='rating'")[0]["n"]
        return {"pairwise": pw, "rating": rt}
