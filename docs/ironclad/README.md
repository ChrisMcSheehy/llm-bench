# Engineering notes

This project is built under a discipline called *Ironclad*, whose short version is: every
non-obvious decision has to survive review by someone with no project context, and a
decision that needs a paragraph of defence is usually the wrong decision.

What that produces, and what lives here:

| File | What it is |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The layer map, the rules dependencies must obey, and a dated record of every structural decision with its reasoning |
| [`LESSONS.md`](LESSONS.md) | Defects that got through, what each one cost, and the enforcement added so it cannot recur. Source comments and tests cite these by name |
| `PROBE-*.md` | Measurements taken against real running servers before the code that depends on them was written — what an endpoint actually reports, what a context rung actually costs |

## What is not here

Design documents (`DESIGN-*.md`) and implementation plans (`plans/*.md`) are kept outside
this repository. Source comments and the probe documents above cite them by filename —
`DESIGN-launcher.md`, `plans/2026-08-04-memory-cost-phase-3.md` and so on. Those references
are accurate; the files simply are not published.

Nothing in this repository depends on them. Every decision they record that shapes the code
is also recorded in `ARCHITECTURE.md` under **Structural decisions**, which is the
authoritative list and is checked by
[`tests/test_docs_match_the_code.py`](../../tests/test_docs_match_the_code.py) — a test that
exists because the same documentation drift shipped twice.

## Where to start

If you want to know what the tests do, read [`../TESTS-EXPLAINED.md`](../TESTS-EXPLAINED.md)
— all 582 of them in plain English.

If you want to know why the code is shaped the way it is, read `ARCHITECTURE.md`.

If you want to know what this project has got wrong, read `LESSONS.md`. It is the most
useful document here.
