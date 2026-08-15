"""Text-to-SQL, graded by execution accuracy (Spider/BIRD flavoured).

A small retail schema is built in a throwaway SQLite database. For each question
the model is given the schema DDL and must produce a SELECT; we execute both the
model's SQL and the gold SQL and compare result sets (order-insensitive). Only
read-only SELECT statements are executed.

This is deterministic, needs no download, and is a realistic probe for anyone
using local models for analytics work.
"""
from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path

from llmbench.evaluators.base import EvalContext, Evaluator, Verdict
from llmbench.models import Metric, Sample
from llmbench.registry import register

_SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, joined TEXT);
CREATE TABLE products  (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
CREATE TABLE orders    (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
                        qty INTEGER, order_date TEXT);
"""
_SEED = """
INSERT INTO customers VALUES (1,'Ada','Leeds','2025-01-05'),(2,'Ben','Bristol','2025-02-11'),
 (3,'Cara','Leeds','2025-03-02'),(4,'Dan','York','2025-03-20');
INSERT INTO products VALUES (1,'Widget','hardware',9.99),(2,'Gadget','hardware',19.50),
 (3,'Manual','media',4.00),(4,'Cable','hardware',3.25);
INSERT INTO orders VALUES (1,1,1,4,'2025-04-01'),(2,1,3,1,'2025-04-02'),(3,2,2,2,'2025-04-02'),
 (4,3,1,10,'2025-04-05'),(5,3,4,5,'2025-04-06'),(6,4,2,1,'2025-04-09'),(7,1,2,3,'2025-04-10');
"""

# (id, question, gold_sql)
_TASKS = [
    ("count_leeds", "How many customers are based in Leeds?",
     "SELECT COUNT(*) FROM customers WHERE city='Leeds'"),
    ("hardware_names", "List the names of all products in the 'hardware' category.",
     "SELECT name FROM products WHERE category='hardware'"),
    ("top_spender",
     "Which customer name has the highest total order value (qty*price)? Return just the name.",
     """SELECT c.name FROM customers c JOIN orders o ON o.customer_id=c.id
        JOIN products p ON p.id=o.product_id
        GROUP BY c.id ORDER BY SUM(o.qty*p.price) DESC LIMIT 1"""),
    ("units_per_category",
     "For each product category, what is the total quantity ordered? Return category and total.",
     """SELECT p.category, SUM(o.qty) FROM products p JOIN orders o ON o.product_id=p.id
        GROUP BY p.category"""),
]

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.DOTALL | re.I)
_DEFAULTS = {"max_tokens": 256, "temperature": 0.0}


def _extract_sql(text: str) -> str:
    blocks = _SQL_BLOCK.findall(text)
    sql = (blocks[0] if blocks else text).strip()
    m = re.search(r"\bSELECT\b.*", sql, re.DOTALL | re.I)
    sql = m.group(0) if m else sql
    return sql.split(";")[0].strip()


@register
class Text2SQLEvaluator(Evaluator):
    name = "text2sql"
    version = "1"
    default_config = _DEFAULTS

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        dbpath = Path(tempfile.mktemp(suffix=".db"))
        conn = sqlite3.connect(dbpath)
        conn.executescript(_SCHEMA + _SEED)
        conn.commit()
        try:
            return [await self._one(ctx, cfg, conn, t) for t in _TASKS]
        finally:
            conn.close()
            dbpath.unlink(missing_ok=True)

    async def _one(self, ctx, cfg, conn, task) -> Sample:
        tid, question, gold = task
        prompt = (f"Given this SQLite schema:\n{_SCHEMA}\n"
                  f"Write a single SQL SELECT query to answer: {question}\n"
                  f"Output only the SQL.")
        def grade(res) -> Verdict:
            pred_sql = _extract_sql(res.text)
            gold_rows, _ = self._run(conn, gold)
            pred_rows, err = self._run(conn, pred_sql)
            ok = err is None and gold_rows is not None and self._match(gold_rows, pred_rows)
            return Verdict(score=1.0 if ok else 0.0, passed=ok,
                           meta={"sql": pred_sql[:300], "error": err})

        return await self.run_case(
            ctx, case_id=tid, group=tid,
            messages=[{"role": "user", "content": prompt}], grade=grade,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"])

    def _run(self, conn, sql):
        if not re.match(r"^\s*SELECT\b", sql, re.I):
            return None, "not a SELECT"
        try:
            cur = conn.execute(sql)
            return cur.fetchall(), None
        except Exception as e:
            return None, str(e)

    def _match(self, gold, pred) -> bool:
        if pred is None:
            return False
        norm = lambda rows: sorted(tuple(str(c) for c in r) for r in rows)
        return norm(gold) == norm(pred)
