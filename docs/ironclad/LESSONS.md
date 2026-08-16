# Lessons

Things this project has already paid to learn. Grep this before writing a design or a
plan; a proposal that repeats one of these is dead on arrival.

`grep "^## "` lists every lesson. Entries from 2026-08-02 onward use the fixed
Trigger/Lesson/Enforcement/Scope form; the two narrative entries below it predate that
and are left as written rather than rewritten from memory.

---

## [2026-08-02] pytest-pythonpath-tests-the-working-tree-not-the-wheel

- **Trigger:** designing the cross-platform check matrix. The obvious shape — "run pytest
  on three operating systems" — would have reported green against a wheel that shipped no
  data at all. `pyproject.toml` sets `pythonpath = ["."]`, so pytest imports `llmbench`
  from the repository root, where `llmbench/data/` sits regardless of what setuptools
  actually packaged. `tests/test_resources.py` looks like it guards packaging. It does not.
- **Lesson:** a suite that puts the source tree on the import path tests the source tree,
  and nothing it asserts about an *installed* artefact means anything. Proving packaging
  needs a built wheel, installed non-editable, exercised from a directory that is not the
  repository — which is a separate job, not another test file.
- **Enforcement:** the `packaging` job in `.github/workflows/checks.yml` builds a wheel,
  installs it, and resolves the bundled data from `runner.temp` on all three platforms.
- **Scope:** global

Related: [[verification-should-exceed-the-plans-minimum]].

---

## [2026-08-02] scrubbed-subprocess-env-needs-systemroot-on-windows

- **Trigger:** the coding harness ran its child with `env={"PATH": ...}` only. On Windows
  that child died during pytest startup on `import asyncio` (`WinError 10106`, Winsock
  cannot initialise), so every coding problem — including the project's own reference
  solutions — scored zero while looking like an ordinary weak result.
- **Lesson:** a deliberately scrubbed environment must still pass `SYSTEMROOT` through on
  Windows; it is required by the OS itself, not by the program.
- **Enforcement:** `_child_env()` in `evaluators/coding.py`, plus
  `tests/test_coding_harness.py::test_the_reference_solution_passes`, which fails if the
  harness cannot run known-good code.
- **Scope:** global

---

## [2026-08-02] assert-the-success-condition-not-the-absence-of-error

- **Trigger:** two instances the same day. The end-to-end test asserted that coding
  *metrics existed* — which stayed true when the harness executed nothing and graded
  everything zero. Separately, a subprocess was timed with `capture_output=True` and its
  output never read, so a hard failure was recorded as a healthy 0.45s.
- **Lesson:** assert the thing that means it worked (a known-good input passes, the
  return code is zero), never the absence of an exception or the presence of a field. A
  check that cannot tell success from failure is not a check.
- **Enforcement:** `selftest.py` now asserts `coding == 1.0`, not merely that coding
  reported something; `tests/test_coding_harness.py` runs the reference solution.
- **Scope:** global

---

## [2026-08-04] rich-substitutes-for-some-unencodable-characters-but-not-all

- **Trigger:** the build sweep printed `── build-a ──` between builds. Run in a terminal
  it was fine; piped on Windows it died with `UnicodeEncodeError: 'charmap' codec can't
  encode characters in position 0-1`, taking the whole run with it. The same stream
  printed a fingerprint label's middle dot and a rich table's borders quite happily,
  degrading them to `?`.
- **Lesson:** rich's substitution is **selective**, so "it printed fine when I piped it"
  proves nothing about the next character the tool learns to print. A redirected stream on
  Windows gets a legacy code page, and any un-substituted character on it is a crash in
  ordinary use, not a cosmetic problem. Note the earlier entry
  [[eyeballed-console-output-is-not-the-string]] treated this family as a *verification*
  hazard; it is also a *runtime* one.
- **Enforcement:** `cli.py` reconfigures `sys.stdout`/`sys.stderr` to UTF-8 with
  `errors="replace"` at import, so the fix is at the entry point rather than per
  character. `tests/test_cli_encoding.py` drives a real subprocess under
  `PYTHONIOENCODING=cp1252`; both its tests were confirmed to fail with the fix disabled.
  An in-process `CliRunner` cannot catch this — it replaces the very streams that are
  misconfigured.
- **Scope:** global. Applies to any output the CLI gains, not only to the rule that
  exposed it.

Related: [[eyeballed-console-output-is-not-the-string]],
[[scrubbed-subprocess-env-needs-systemroot-on-windows]].

---

## [2026-08-04] observe-the-real-thing-rather-than-assert-something-unfalsifiable

- **Trigger:** writing the launcher's "a server that never becomes ready is cleaned up"
  test. `start()` cleans up internally and then raises, so the test has no handle on the
  process it wants to check. The first draft papered over that with a helper that returned
  `False` — an assertion that could never fail, dressed as a cleanup check.
- **Lesson:** when the code under test cleans up after itself, there is nothing left to
  assert against, and the tempting move is to assert something adjacent that is always
  true. The result passes forever and proves nothing — worse than no test, because it
  occupies the space where the real one would go.
- **Enforcement:** observe the real object instead of inventing a proxy. Here, a fixture
  wraps `subprocess.Popen` to record every process actually created, so the test asserts
  `poll() is not None` on genuine processes. Wrapping the real constructor keeps the child
  real; only the bookkeeping is added. Before writing any assertion, ask what would have to
  break for it to fail — if nothing can, it is not a test.
- **Scope:** global.

Related: [[assert-the-success-condition-not-the-absence-of-error]],
[[unit-tests-either-side-of-a-seam-do-not-test-the-seam]].

---

## [2026-08-04] unit-tests-either-side-of-a-seam-do-not-test-the-seam

- **Trigger:** Phase 2 (D6) added four launch-argument fields to the identity, shipped 21
  green tests, and went green on a twelve-job three-OS matrix. The first probe against a
  real `llama-server` showed all four fields arriving as `None`: the adapter reads argv
  from `/v1/models` → `data[0].status.args`, and `status` exists only when the server runs
  in **router** mode. In the ordinary `-m model.gguf` deployment the field is absent, so
  `-ngl 40` and `-ngl 99` still hash identically — the exact bug the phase existed to fix.
- **Lesson:** the tests covered `_parse_args(argv) → dict` and `ModelFingerprint(fields) →
  hash`, both real. Nothing covered *where argv comes from*. Testing each side of a seam
  and never the seam itself produces total green over a feature that cannot fire, and the
  greener the suite the more confidently the gap ships.
- **Enforcement:** a detection test must exercise `detect()` against a served payload —
  a recorded fixture of a real `/props` + `/v1/models` response is enough and needs no
  live server. Any field sourced from a backend response needs one, and the fixture must
  come from a **capture**, never from a hand-written guess at the shape.
- **Scope:** global. Applies to every adapter field, not only to these four.

Related: [[assert-the-success-condition-not-the-absence-of-error]],
[[pytest-pythonpath-tests-the-working-tree-not-the-wheel]] — same failure in a different
coat: a suite proving something about an artefact it is not actually exercising.

---

## [2026-08-04] a-launch-argument-is-a-request-not-a-fact

- **Trigger:** the same probe. `llama-server` was started with `-b 2048`; its own log then
  recorded `embeddings enabled with n_batch (2048) > n_ubatch (512) … setting n_batch =
  n_ubatch = 512`. The argv says 2048. The server ran 512.
- **Lesson:** parsed command-line arguments describe what was *asked for*. Anything that
  files results under them is filing them under an intention rather than a configuration —
  and for an identity hash, that means two genuinely different runs can share a hash while
  carrying a label that describes neither.
- **Enforcement:** prefer a server-reported value over a parsed argv value wherever both
  exist (`/props.total_slots` over `-np`, `default_generation_settings.n_ctx` over `-c`).
  Use argv only where nothing else reports the setting, and treat that as a known weakness
  rather than a source of truth.
- **Scope:** global — applies to every backend adapter.

Related: [[unit-tests-either-side-of-a-seam-do-not-test-the-seam]].

---

## [2026-08-04] eyeballed-console-output-is-not-the-string

- **Trigger:** executing Phase 2's T2.4. Its verification prints a label and gives the
  expected text `Qwen3-8B · Q4_K_M · kv:q8_0 · ngl:99 · b:4096`. On this Windows console
  the middle dots came out as `?` — the label was byte-for-byte correct, but the printed
  form did not match the plan's expected output.
- **Lesson:** a verification that compares printed output *by eye* also tests the
  terminal's encoding, which is not the thing under test. Where the expected value
  contains any non-ASCII character, the check reads as a failure on a console that cannot
  render it — and the tempting "fix" is to change the code that was already right.
- **Enforcement:** make the comparison inside the process and print the verdict:
  `print('match:', got == want)` plus `ascii(got)` for the diff, rather than printing
  `got` and comparing with your eyes. Plans should write verification steps this way when
  the expected string is not pure ASCII.
- **Scope:** global

Related: [[assert-the-success-condition-not-the-absence-of-error]],
[[verification-should-exceed-the-plans-minimum]].

---

## [2026-08-04] validate-an-estimate-against-the-thing-it-estimates

- **Trigger:** Phase 3 shipped a KV-cache estimate that was checked three ways — unit tests
  on hand-built shapes, a real GGUF file parsed end to end, and a figure matching the probe
  document to the byte. All three compared the arithmetic against **its own inputs**.
  Raising llama-server's verbosity for an unrelated reason showed that the server prints
  what it actually allocated: `size = 128.00 MiB (4096 cells, 8 layers, 2/2 seqs)`.
  Against a real server the estimate was **448 MiB where the truth was 1088 MiB** — two
  independent errors, neither visible to any check that existed.
- **Lesson:** when the thing you are estimating is allocated by a real system, that system
  usually states the true value somewhere — a log line, a metric, a status endpoint. Until
  you have compared against it at least once, you have verified self-consistency and
  called it correctness. The more thoroughly the formula is tested against its own
  assumptions, the more confident the wrong number becomes. Before shipping a derived
  figure, spend one run looking for where the subject says the answer out loud.
- **Enforcement:** `tests/test_memory.py` now carries the server's own figures as
  assertions — `test_it_matches_what_the_server_allocated_for_a_split_cache` (1088 MiB)
  and `..._for_a_unified_cache` (1568 MiB) — so the formula is pinned to a real
  allocation rather than to its own inputs, and any future change to it must still
  reproduce both. [[../PROBE-2026-08-04-host-facts.md]] Finding 5 records the log lines
  and the five runs those numbers came from, so the assertions can be re-derived rather
  than trusted.
- **Scope:** global.

Related: [[observe-the-real-thing-rather-than-assert-something-unfalsifiable]],
[[a-reported-figure-can-describe-a-slot-rather-than-the-whole]] — the same family as both:
a check that never touches the real object cannot report on it.

---

## [2026-08-04] a-reported-figure-can-describe-a-slot-rather-than-the-whole

- **Trigger:** Phase 3 sized the KV cache from `/props.default_generation_settings.n_ctx`,
  having preferred that server-reported value over the `-c` launch argument — the right
  instinct, per [[a-launch-argument-is-a-request-not-a-fact]]. A probe against a server
  started with `-c 8192 -np 2` then reported `n_ctx = 4096`: the figure is **per slot**.
  The shipped estimate is half the true cache on that server, and in general too small by
  `total_slots` whenever the cache is not unified. The plan had flagged the risk as an
  open question and guessed the error in the opposite direction, then shipped anyway.
- **Lesson:** preferring the server's own number protects you from a stale *request*, but
  not from a **different question**. A scalar that could be a total or a per-unit share is
  ambiguous until something observed resolves it, and the ambiguity is invisible on the
  single-unit configuration where both readings coincide — which is the one you develop
  against. Before using a reported scalar in arithmetic, find a configuration where total
  and per-unit differ, and check which one you are being given.
- **Enforcement:** [[../PROBE-2026-08-04-host-facts.md]] Finding 4 records the two runs and
  the flag (`-kvu` defaults to on only when the slot count is auto). The README carries the
  gap explicitly until the correction is designed. The correction itself is a design
  decision — `kv_unified` is absent from every HTTP endpoint, so the factor is knowable
  only from arguments llmbench supplied itself.
- **Scope:** global.

Related: [[a-launch-argument-is-a-request-not-a-fact]],
[[unit-tests-either-side-of-a-seam-do-not-test-the-seam]] — the seam here is not code-to-
code but number-to-meaning, and no test on either side of it could see the mismatch.

---

## [2026-08-04] a-captured-fixture-carries-paths-that-still-exist-at-home

- **Trigger:** Phase 3's T3.4. The plan's test asserted that detection reports an unknown
  memory figure for the captured `/props` payload, on the stated grounds that "the
  payload's `model_path` points at a machine that is not this one". It points at *this*
  one — the payload was captured here — so the file opened, the shape was read, and
  detection correctly returned 75497472 where the test demanded `None`. The test would
  have passed on all twelve CI jobs and failed only on the machine that wrote it.
- **Lesson:** a captured fixture carries real absolute paths from the capture machine, and
  an assertion that depends on one being **absent** is an assertion about the machine
  rather than about the code. It fails where the fixture came from and passes everywhere
  else — and once the file is eventually deleted it passes everywhere, still proving
  nothing. Construct the unreachable resource instead: `tmp_path / "absent.gguf"` is
  guaranteed missing on every machine, forever.
- **Enforcement:** `tests/test_detect_from_server.py::test_detection_reports_unknown_when_the_model_file_is_not_reachable`
  overwrites `model_path` with a path under `tmp_path` before detecting, and its docstring
  says why. Any test of a "resource unreachable" branch must build the missing resource,
  never inherit its absence from the environment.
- **Scope:** global.

**Recurrence — 2026-08-15, the same lesson from the opposite side.** The first run of the
three-operating-system matrix after 2026-08-04 turned up
`test_declared_settings.py::test_a_declared_unified_flag_resolves_a_multi_slot_server`
failing on **all twelve jobs** while passing locally. It called `_detect` without a model,
so the fixture's captured `model_path` stood —
`C:/Users/chris/AppData/Roaming/.../nomic-embed-text-v1.5.Q4_K_M.gguf` — which still exists
on the capture machine. At home the shape was read and a figure came back; everywhere else
the shape was unknown and the figure was `None`.

The original enforcement above covers a test that needs a resource **absent**. This one
needed it **present**, and the wording did not reach it. The rule is therefore widened:
**a test must construct every resource whose presence or absence decides its verdict** —
in either direction. Inheriting either state from the machine makes the machine the
examiner. Fixed by `_dense_model()` in that file, which builds its own model.

Note what made this visible at all: the defect shipped on 2026-08-04 and sat green for
eleven days because the matrix was not running — see
[[a-check-that-stops-running-looks-nothing-like-a-check-that-fails]], which is not a
separate story from this one but the reason this one lasted.

Related: [[observe-the-real-thing-rather-than-assert-something-unfalsifiable]],
[[pytest-pythonpath-tests-the-working-tree-not-the-wheel]] — all three are a check whose
verdict is decided by the machine it runs on rather than by the code under test.

---

## [2026-08-04] a-permissive-model-swallows-the-field-a-red-test-needs

- **Trigger:** Phase 3's T3.3. The plan predicted two failing tests before the fingerprint
  gained its new fields, and attached a stop-and-report to the case where only one failed:
  "the field already exists". One failed. The field did **not** exist — pydantic's default
  `extra="ignore"` silently drops an unrecognised constructor argument, so a test that only
  *passes* `kv_cache_bytes=…` and then compares two hashes cannot fail before the field is
  added. It passed vacuously, for a reason unrelated to the plan's diagnosis.
- **Lesson:** a red-phase test that exercises a new field only through a permissive
  constructor proves nothing while the field is missing; it must **read the attribute
  back**. And a plan's stop-and-report trigger carries a diagnosis that can itself be
  wrong — confirm the stated cause against the library's actual behaviour
  (`Model.model_fields`) before either stopping or continuing on its say-so.
- **Enforcement:** this entry, plus the habit it names — the field's absence was confirmed
  with `'kv_cache_bytes' in ModelFingerprint.model_fields` before proceeding, and that
  one-line check is the cheapest way to settle any "does this field exist yet" question on
  a pydantic model.
- **Scope:** global.

Related: [[assert-the-success-condition-not-the-absence-of-error]],
[[observe-the-real-thing-rather-than-assert-something-unfalsifiable]].

---

## [2026-08-05] a-truncated-answer-is-not-a-wrong-answer

- **Trigger:** the first run against a real model. Every needle rung scored **0.00** at
  every context length, including 4096, which a 12B model recalls trivially. The samples
  recorded `answer: ''` with `output_tokens: 32` — the model had generated a full budget of
  tokens and llmbench had read the text as empty. Asking the endpoint directly explained it:
  gemma4-heretic is a reasoning model, so `choices[0].message.content` was `""` while
  `reasoning_content` held its thinking, and `finish_reason` was `"length"`. It had spent the
  32-token answer budget reasoning and never reached an answer.
- **Lesson:** *"the model produced no usable answer"* and *"the model answered wrongly"* are
  different facts, and the grader collapsed them into the same 0.00. This is the exact
  category error D3 was written to remove — a gap recorded as a figure — reappearing one
  layer down, in generation rather than in the ladder. It is worse here than a missing
  figure, because a skipped rung is visibly absent whereas 0.00 is a confident measurement
  that a reader will believe. Every quality number this bench produces for a reasoning model
  is currently 0.00 and looks like genuine incapacity.
- **Contributing defect:** `GenResult.truncated` is populated from `data.get("truncated")` —
  a **top-level key that an OpenAI-compatible response does not have**. The real signal is
  `choices[0].finish_reason == "length"`, so `truncated` has been `None` on every sample ever
  recorded. The one field that could have explained the zero was silently never set. Written
  against a remembered response shape and never checked against a live one, which is the same
  root as [[validate-an-estimate-against-the-thing-it-estimates]].
- **Enforcement:** a sample whose generation stopped on a token limit, or whose content is
  empty while the model emitted tokens, must not be graded — it is `skipped` with a reason,
  or an `error`, never a score. Read `finish_reason` from the choice; treat empty `content`
  with non-zero `completion_tokens` as an unusable response and say which it was.
- **Scope:** global to this project. Applies to every evaluator, since all of them grade text
  returned by `Target.generate`.

Related: [[observe-the-real-thing-rather-than-assert-something-unfalsifiable]] — a mock that
always answers correctly cannot show you the case where the answer never arrives.

---

## [2026-08-05] listening-is-not-ready

- **Trigger:** with detection fixed to fail loudly, `llmbench run --server heretic-12b`
  failed at detection every single time: *503 Service Unavailable for /props*. The
  launcher polls `/props` until something answers and then reports the server ready, and
  its readiness helper said so explicitly — *"Answering at all means the server is up; it
  is entitled to dislike the request."* llama-server binds its port immediately and
  answers 503 for as long as it takes to load the weights, which for an 11.8 GB model is
  about ten seconds.
- **Lesson:** accepting a connection and being able to serve are different states, and a
  readiness check that cannot tell them apart declares victory at the earliest possible
  moment — the one guaranteed to be wrong. The rationale here was not an oversight but a
  written-down argument, and it was right about 404 and 401 (the server is up and
  dislikes *this request*) while being wrong about 503 (the server is up and can serve
  *nothing yet*). Collapsing two meanings into "it answered" is what made it wrong.
- **Enforcement:** a readiness probe must name the states it accepts rather than
  accepting whatever is not an exception. Here: 503 keeps waiting, every other status is
  ready. `tests/test_launcher_readiness.py` drives the real helper against a real HTTP
  server that answers 503 then 200, because a mocked one would only prove the mock agrees
  with itself.
- **Scope:** global. Applies to any wait-for-ready loop over HTTP.

Related: [[detection-that-swallows-its-failures-invents-an-identity]] — these two
defects were the same incident seen from either end, and the first one hid the second for
three days. A loud failure is what let the real cause be found in minutes.

---

## [2026-08-05] detection-that-swallows-its-failures-invents-an-identity

- **Trigger:** `llmbench detect` against a real `llama-server` immediately after launching it
  returned a complete fingerprint of nothing — `model_id: "unknown"`, `n_ctx: null`, no build
  number — and hashed it into `fdaa92e6cab29d3f`, which looks exactly like a real identity.
  Run again moments later it returned the truth: build 10144, commit `d73c1d6b2`, `n_ctx`
  32768, 4 slots, hash `873ef11872c7f543`. The stored database then revealed the same
  all-empty fingerprint had been recorded once before, on **2026-08-02**, and nobody noticed.
- **Mechanism, confirmed 2026-08-05 once the fix made it visible:** llama-server binds
  its HTTP port before the weights are loaded and answers `/props` with **503 Service
  Unavailable** until they are. The launcher treated any HTTP answer as ready, so
  `llmbench launch` returned during the load, and detection then ran against a server
  that could not yet describe itself. The swallowed 503 is where `fdaa92e6cab29d3f` came
  from. Fixing the swallow turned a silent corruption into an accurate error message,
  and the error message is what identified the cause - see [[listening-is-not-ready]].
- **Cause:** `Target._get` is `except Exception: return None`, and `detect()` builds a
  fingerprint out of defaults from whatever it got. A failed probe is therefore
  indistinguishable from a server that genuinely reports nothing, and the result is not an
  error but a *plausible-looking identity* under which real results get filed.
- **Lesson:** identity is the foundation this project rests on — every comparison, every
  pooled average, every stored vote is keyed by that hash. A detection path that degrades to
  defaults instead of failing converts a transient fault into permanent bad data, and does it
  silently. "Missing data fails loudly rather than quietly" was Phase 1's stated principle;
  this is the one place it was not applied, and it is the place it mattered most.
- **Enforcement:** `detect()` must distinguish *the server said nothing* from *we could not
  ask*. A probe that fails to reach the server is an error the caller sees, not an empty
  dict. A fingerprint with no `model_id` and no `n_ctx` should never be written to the store
  at all.
- **Scope:** global to this project. `_get` is on the shared base class, so every backend
  adapter inherits the behaviour.

Related: [[a-check-that-stops-running-looks-nothing-like-a-check-that-fails]] — silence
mistaken for a result, in both cases.

---

## [2026-08-05] a-running-server-serves-the-code-it-started-with

- **Trigger:** Phase 6 task E6 added a per-cell sample count to `Store.needle_heatmap` and
  wired it into the heatmap's hover text. `pytest` was green, so the next step was to look
  at the real page — and the rendered Plotly trace had no `customdata` at all. The obvious
  reading was that the frontend edit had not taken, and the next move would have been to
  rewrite working JavaScript. The actual cause: the dashboard had been started with
  `uvicorn` **before** `store.py` was edited, and a Python process holds the module it
  imported. The API was still returning the old shape, without `n`, from a version of the
  file that no longer existed on disk.
- **Lesson:** a long-running process verifies the code it started with, not the code in the
  working tree. Every editor-and-server workflow has this hazard, and it is at its most
  misleading exactly when it matters most — during a manual "look at it" check, which is
  reached precisely because the automated checks already passed. The symptom points at the
  layer you edited last (here, the browser), never at the process you forgot to restart.
- **Enforcement:** when a manual check contradicts a passing test, **check the server's
  age before changing any code**. The cheapest discriminator is to ask the API directly
  rather than the page — `fetch('/api/...')` for the field you just added. If the endpoint
  does not have it, the process is stale; if it does, the bug is really in the frontend.
  Restart on any Python edit; uvicorn's `--reload` is the standing fix and is worth passing
  whenever a server is left up across edits.
- **Scope:** global. Applies to any process serving code that is being edited underneath it.

Related: [[eyeballed-console-output-is-not-the-string]] — both are about the gap between
what you are looking at and what the program actually does.

---

## [2026-08-04] a-check-that-stops-running-looks-nothing-like-a-check-that-fails

- **Trigger:** Phase 5 was about to run its three-operating-system matrix. Checking first
  showed the last **five** pushes had never run at all: every job was blocked before
  starting with *"The job was not started because recent account payments have failed or
  your spending limit needs to be increased."* The block began at 19:43 on 2026-08-04 and
  every push since was affected — including both of Phase 4's, whose plan step said
  "Expected: twelve jobs green". Phase 4 is the phase that added `hostinfo.py`: `os.sysconf`
  on Unix against `ctypes.windll.GlobalMemoryStatusEx` on Windows, `Get-CimInstance` against
  `nvidia-smi`, and a `--list-devices` subprocess. The most platform-divergent code in the
  project shipped with **zero** cross-platform verification, and the plan that required
  that verification was marked done.
- **Lesson:** a suite that fails puts a red wall in front of you. A check that *stops
  running* produces silence, and silence is indistinguishable from success to anyone not
  actively looking. Worse, a blocked job is not shaped like a broken one — twelve jobs
  "failing" in two to nine seconds with no logs reads as flaky infrastructure rather than
  as no coverage at all, so even a glance can wave it away. Any verification that lives
  outside the artefact — a hosted matrix, a nightly, a dashboard someone checks — can be
  skipped without leaving a mark in the repository, and a plan step that says "expected:
  green" cannot tell "I saw green" from "I did not look".
- **Enforcement:** a claim about a remote check must carry the **evidence, not the verdict**
  — the run id and its conclusion, pasted into the report. `gh run list --limit 1 --json
  headSha,conclusion` against your own HEAD takes one command and cannot be satisfied by
  assertion. This is deliberately weaker than the enforcement
  [[verification-should-exceed-the-plans-minimum]] eventually got, and the reason is worth
  stating: that one became `tests/test_docs_match_the_code.py` because the property was
  local and offline, whereas a test that queried CI would need network and credentials,
  would fail for everyone working on a train, and would be circular when run *by* CI. So
  this one stays a discipline, and this entry is the record of what it costs when the
  discipline lapses.
- **Scope:** global. Applies to every check the repository does not run itself.

Related: [[verification-should-exceed-the-plans-minimum]],
[[assert-the-success-condition-not-the-absence-of-error]] — the same family seen from a
third angle: here the check could not tell success from *never having been asked*.

---

## [2026-08-02] empty-registry-is-not-an-initialised-flag

- **Trigger:** `registry.get()` ran discovery only `if not _REGISTRY`. Any code importing
  a single evaluator module first registered exactly that one, after which the registry
  read as populated and the other ten were never discovered — the suite then failed with
  "No evaluator named 'mcqa'".
- **Lesson:** never use "the container is empty" as the flag for "initialisation has
  run". Partial population is the case that breaks it, and it arrives from callers you
  do not control.
- **Enforcement:** explicit `_discovered` flag in `registry.py`, plus
  `tests/test_registry.py`, which does the partial import in a fresh interpreter.
- **Scope:** global

---

## [2026-08-16] a-key-that-is-unique-in-a-small-set-stops-being-unique-in-a-large-one

- **Trigger:** importing HumanEval took the coding problem set from 4 directories to 168.
  `selftest.py`'s mock backend answers a coding question with that problem's own reference
  solution, and found it by scanning `problem.yaml` files for the entry-point name **as a
  substring**. With four problems the names were distinct and it worked. With 168 it
  returned the wrong problem's solution for **twenty** of them, and six entry-point names
  turned out to be shared by two HumanEval problems each, so even an exact match on the
  entry point would not have been unique. The end-to-end test failed with
  `coding=0.8869`, and its own comment says anything below 1.0 means *the harness failed
  to run correct code* — so the first reading was a fault in the code-execution harness,
  which is the most alarming thing this project has.
- **Lesson:** a lookup key is only as unique as the data it has met so far, and "it has
  always worked" is evidence about the old size of the set, not about the key. The
  dangerous part is not the collision, it is the direction of the failure: handing back a
  *correct solution to a different question* produces a plausible score, not an error, and
  the failure surfaces in whatever the wrong answer breaks — here, as an accusation
  against the harness. A test fixture that identifies things loosely will eventually
  slander the code it was written to protect.
- **Enforcement:** the mock now keys on the **prompt**, which appears verbatim in the
  message the evaluator builds and is genuinely unique, and it is built once and sorted
  longest-first so that one prompt containing another cannot shadow it. The lookup is a
  named function with the reasoning in its docstring rather than an inline scan.
- **Scope:** global. Applies to every match-by-substring and every lookup keyed on
  something merely *usually* distinct — names, labels, prefixes, titles.

Related: [[a-substring-match-discards-the-qualifier-that-made-it-different]] — the same
mistake seen from the other end: there, extra characters around the match were discarded;
here, the match was found in the wrong place entirely.

---

## [2026-08-15] a-substring-match-discards-the-qualifier-that-made-it-different

- **Trigger:** reading Unsloth's documentation while answering an unrelated question.
  `parse_quant` searched a model filename for a known quantisation token, so
  `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` and `Qwen3.6-35B-A3B-Q4_K_M.gguf` both returned
  `Q4_K_M`. The `UD-` prefix is not a spelling variant: an Unsloth Dynamic quant picks
  the type per layer, and their own measurements put Dynamic "Q4" near uniform Q5 for
  perplexity. The two files are the same nominal size and different programs, and `quant`
  is the field this entire bench exists to compare. This project's own launch-profile
  example in `DESIGN-benchmark-coverage.md` names a `UD-` model, so the collision was
  already sitting in the documented workflow.
- **Lesson:** a parser that *searches* a longer name for a known token silently discards
  whatever qualified it. The match succeeds, the value looks exactly right, and nothing
  anywhere fails — the defect is only ever visible by comparing the input to the output,
  which is precisely what a passing test suite gives nobody a reason to do. Extraction by
  substring is a claim that everything around the substring is noise, and that claim
  should be written down and tested rather than assumed by a regex.
- **Enforcement:** the prefix is part of the pattern in `models.py`, and
  `tests/test_identity.py` pins all four directions: a Dynamic quant keeps its prefix, a
  stock quant does not gain one, a word merely ending in "ud" is not a Dynamic quant, and
  the two hash to different fingerprints. The negative cases matter as much as the
  positive one — a rule that labelled everything `UD-` would also have removed the
  collision and been wrong about every other model on disk.
- **Scope:** global. Applies to every parser in the project that reaches into a name for a
  known token: `parse_params` and `model_name_from_path` are the same shape.

Related: [[a-launch-argument-is-a-request-not-a-fact]] — both are about a value that reads
as a fact about the deployment while actually being a fact about a string.

---

## moving-shared-files-needs-a-grep-not-a-memory

**2026-07-26 — cost: one red test run, two mid-execution plan amendments.**

A plan moved the bundled data directories (`datasets/`, `problems/`, `suites/`) into the
package. Its task listed the six evaluator files that referenced those paths, enumerated
from the module structure.

Two further references existed and were missed:

- `selftest.py` held its own module-level `Path("problems/coding")`. The end-to-end test
  went red, and the symptom — "missing metrics for coding" — pointed at the coding
  evaluator rather than at the test harness.
- `cli.py` defaulted its suite argument to the relative string `"suites/default.yaml"`.
  Nothing failed in the test suite; the break only appeared when running the installed
  command from an unrelated directory, which happened to be a verification step. Had that
  step not existed, this would have shipped.

**Why it happened:** the file list was written from knowledge of which *modules* consume
data, not from a search for which *files* mention the paths. Entry points and test
harnesses consume data too, and neither is a module in the mental map.

**How to apply:** before writing a task that moves or renames a shared resource, grep the
entire tree for every literal that mentions it, and paste the grep output into the plan
as the file list. Enumerating from memory of the architecture will miss the CLI, the test
harness, and anything else that is not a "real" module.

Related: [[verification-should-exceed-the-plans-minimum]].

---

## verification-should-exceed-the-plans-minimum

**2026-07-26 — cost: nearly shipped a stale instruction in the README.**

A task's verification checked two things about the README: that the old install command
was gone, and that the new database location was documented. Both passed.

A third assertion, added while executing because it was free, failed — a second copy of
the stale `llmbench run suites/default.yaml` was still present further down the file, in
a section the task had not mentioned.

**Why it matters:** the plan's checks were written against the *edits the task made*, so
they could only confirm those edits happened. They could not detect an instance of the
same problem somewhere the task's author had not looked. A check derived from the edit
list will always be blind in exactly the places the edit list is blind.

**How to apply:** write verification against the *property that should hold across the
whole artefact* ("no stale relative suite path anywhere in the README"), not against the
lines the task edited ("line 70 now reads X"). When executing, if a stronger check is one
line away, write it.

**Escalated 2026-08-04 — this recurred twice more.** Phase 3 and Phase 4 each shipped a
plan whose documentation check asked "does the new thing appear in the README?", each
time it did — in the new section — and each time the architecture sketch further down
still described an older program. Two more occurrences of one lesson is an enforcement
failure, not bad luck, so the rule is now a test rather than a habit:
`tests/test_docs_match_the_code.py` asserts that every module and every CLI command
appears in the README sketch and in `ARCHITECTURE.md`'s layer table. Confirmed
falsifiable by removing one of each. A plan can no longer forget this, because the suite
fails.

Related: [[moving-shared-files-needs-a-grep-not-a-memory]].
