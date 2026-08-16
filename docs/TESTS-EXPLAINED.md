# What the tests actually check — in plain English

Written for someone with no knowledge of this project, this codebase, or Python.

## First, what this project is

`llmbench` is a **test bench for AI language models that run on your own computer**.

You have a model file on disk. You run a program (`llama-server`) that loads it and
answers questions. There are dozens of knobs on that program — how much of the model to
put on the graphics card, how much conversation history to allow, how hard to compress
its memory. Every knob changes both the **quality** of the answers and the **speed**.

llmbench's job is to answer *"if I turn this knob, what did it cost me?"* It asks the
model a batch of questions, marks the answers, times them, and files everything away so
you can compare setups side by side.

## Why this project has so many tests

The whole product is **a number you are supposed to trust**. If the bench says setup A
scores 0.83 and setup B scores 0.79, you are going to act on that.

So almost every test here is guarding one of three things:

1. **Is the bench filing this result under the right setup?** If it can't tell setup A
   from setup B, it will silently average them together and both numbers become fiction.
2. **Is this number honest?** A question that was never asked must never be counted as a
   question that was failed.
3. **Does the thing actually happen?** Not "did the code run without complaining", but
   "did the server really start, did the process really die, was the answer really
   graded".

A striking feature of this test suite: **most files begin by describing a real bug that
really happened**, with the date. These are not hypothetical tests. Each one is a scar.

## How to run them

```bash
.venv/Scripts/python.exe -m pytest -q
```

(On macOS or Linux the path is `.venv/bin/python`.)

Current state: **47 test files, 378 checks, all passing** — verified 2026-08-15, 75
seconds.

---

# Group 1 — "Do we know what we are testing?"

This is the foundation. Every result is filed under a **fingerprint**: a short code
computed from all the settings that affect the outcome. Change a setting that moves the
numbers, and you get a different fingerprint, so the new results are filed separately
instead of being blended into the old ones.

### `test_identity.py` (15 checks) — what counts as a different setup

Proves the fingerprint changes when it should and *doesn't* when it shouldn't.

- Putting half the model on the graphics card vs. all of it → **two** setups. Correct;
  that changes everything.
- Changing batch settings → different setups, and the three different batch settings are
  told apart from each other rather than lumped together.
- Moving the server to a different network address → **same** setup. Correct; the port
  number doesn't change how good the answers are.
- The memory estimate doesn't change the identity — it's something we *calculated*, not
  something that changes results.
- "Nobody set this knob" and "this knob is set to zero" are different things.
- A **compressed model file's compression scheme** counts, not just how small it is.
  Some publishers vary the compression layer by layer, and such a file is named
  `UD-Q4_K_M` where an ordinary one is `Q4_K_M`. They come out roughly the same size and
  behave quite differently — by the publisher's own figures the clever one at "Q4" is
  about as accurate as a plain "Q5". The bench used to read the name, spot the familiar
  `Q4_K_M` inside it, and file both under that — so the two things you were comparing
  appeared in the table under one identical label. These prove the prefix survives, that
  an ordinary file doesn't wrongly acquire one, that a folder called `cloud-` doesn't
  trip the rule, and that the two now count as two setups.

### `test_binary_identity.py` (6 checks) — which *executable* produced this

The point of the project is comparing builds of llama.cpp — forks, branches, pull
requests. The fingerprint used to record only the commit ID the server reports.

**The bug:** two builds of the *same* commit — say one compiled for Vulkan, one for AMD's
ROCm — looked identical to the bench. It would pool them together, averaging away exactly
the difference the test was measuring. (Observed for real on 2026-08-05: the author's own
machine had a Vulkan build on PATH while Ollama drove the same card through ROCm.)

The fix: take a fingerprint **of the executable file itself**. These tests prove:

- Two different builds of one commit are now two setups.
- The *same* executable copied to a second folder is still **one** setup — because it
  hashes the file's contents, not its path.
- A server we merely connected to (so we don't know its executable) still works; it just
  records "unknown" rather than inventing something.

### `test_detect_from_server.py` (21 checks) — reading the setup off a live server

The biggest file, and it exists because of an embarrassing near-miss recorded in the
project's lessons file:

> A previous phase added four settings to the fingerprint and shipped **21 passing
> tests**, none of which actually ran the detection code end to end. Against a real
> server all four fields came back empty, because llama.cpp only reports them in a
> special mode nobody uses. The tests on either side of the gap couldn't see the gap.

So these tests feed the detection code **real captured responses** from a real
`llama-server`, not hand-written imitations. They check:

- On an ordinary server that reports nothing about its launch settings, the bench admits
  it doesn't know — and such a run is **never pooled** with one where the settings are
  known.
- When llmbench started the server itself, it knows the settings *because it supplied
  them*, and the identity is complete.
- Where the server *does* report something, the server wins over what we asked for.
  (A launch argument is a request; the server may refuse or rewrite it.)

### `test_detection_failure.py` (5 checks) — a broken connection is not a valid answer

**The bug, found 2026-08-05:** pointing the bench at an unreachable server produced a
perfectly-formed fingerprint of *nothing* — model "unknown", no settings — and hashed it
into a code that looks exactly like a real one. Worse, one of these empty ghosts had been
sitting in the results database since 2026-08-02, unnoticed.

These tests prove that an unreachable or uncooperative server is now a **loud error**,
not a silent fake identity. A momentary network fault should not leave permanent garbage
in your results.

### `test_launch_args.py` (2 checks) — reading a command line

Small and mechanical: given a command line like `-ngl 99 -c 16384 -fa on`, pull out the
settings that matter, and understand both the short and long spelling of each flag
(`-b` and `--batch-size` are the same thing).

### `test_declared_settings.py` (7 checks) — telling the bench what it cannot see

Two settings are reported by *no* endpoint anywhere, and both affect the memory
calculation. When you didn't start the server yourself, the honest answer is "unknown".

So you're allowed to **declare** them by hand. The rule these tests enforce is subtle and
important: a declaration is a *claim*, not an *observation*. It's allowed to feed the
memory estimate (which is a calculation that records its own inputs) but it must **never**
touch the identity hash — because the identity must only ever record what was actually
seen. Otherwise you could type your way into a fake distinction between two setups.

---

# Group 2 — "Is this number honest?"

### `test_answer_rate.py` (21 checks) — "how often it was right" needs "how often it spoke"

Two setups both score 85%. One answered every question; the other quietly declined one in
five and scored 85% on the rest. **Those are very different setups, and the leaderboard
showed them as identical rows.**

An earlier fix (see below) stopped a non-answer being marked *wrong*, which was right — but
it left the non-answers invisible. So now every setup also reports **how much of the test
it actually attempted**, printed next to the score it qualifies.

The hard part is what counts in the denominator, and most of these checks are about that:

- A question the model was asked and **said nothing to** counts against it.
- A test the machine **could never run** — a memory limit, a context too large — does
  **not**. Counting it would report a laptop's honest limit as a model refusing to answer.
- A **failed connection** doesn't count either. That's a fact about the network.
- Asked nothing at all, the figure is **absent rather than zero**, because a rate over no
  questions isn't zero.
- Every test module that talks to a model reports it, checked one module at a time, with
  the usual guard that a module added later must be added here too.
- The figure **pools across machines**, unlike speed: a model that spends its budget
  thinking and returns nothing does so on any computer.
- It survives being written to the database, and an older database **gains the column
  without inventing values** for rows recorded before anyone was tracking this.

### `test_reassembly.py` (20 checks) — "it broke" is not the same as "how badly"

Every other long-context test in this project answers in whole numbers: the model either
found the hidden thing or it didn't. That's fine for *did it work*, and useless for the
question this project actually exists to answer — **how much quality did compressing the
memory cost me?** A pass/fail says a line was crossed somewhere between two settings, and
nothing about how far past it you are.

So this test hides three labelled pieces of a long random key at three depths in a big
document, asks for them back joined together, and marks the answer four ways: how many
pieces came back, whether they were in the right order, **how many of the individual bits
are right**, and whether the whole thing is perfect.

The bit score is the point, and it's why the key is written in hex — each character is
exactly four bits, so "90% correct" means something precise. These checks include:

- One mistyped character scores **just under perfect**, not zero. Every other test here
  would call that a flat failure, indistinguishable from finding nothing at all.
- An answer of the **wrong length** reports the bit score as **unknown — a dash, never a
  number**. Comparing bits between different-length strings measures alignment rather than
  memory, and would turn a structural failure into a respectable-looking ~50%.
- But that answer is **still counted as a failure** elsewhere, and the count of pieces
  found still reports what the model retrieved, so the result is diagnosable rather than
  blank.
- The **right pieces in the wrong order** is recorded as an assembly failure with perfect
  retrieval — reading worked, writing didn't. No other test here can express that.
- The key is **generated, never borrowed** from anything real: a genuine published key
  might sit in the model's training data, and a model reciting one from memory would score
  perfectly while retrieving nothing. Same seed, same key, so runs stay comparable — and a
  **different key for every cell**, so a server reusing its memory between questions can't
  carry an answer forward.

### `test_speed.py` (10 checks) — one number that was two numbers in a trenchcoat

**The bug, and it had been published all along.** A model server does two different jobs:
it **reads** your question — all of it at once, as fast as the machine can calculate — and
then it **writes** the answer one word at a time, as fast as it can shuffle the model
through memory. These are different speeds with different limits.

The bench published one number for both: how many words it wrote, divided by the total
time including reading. Then it averaged that across questions of wildly different
lengths. The result moved mostly according to how long the questions were — the one thing
the number never told you. Two identical setups, tested with different question lengths,
looked like different models.

The correct figures were already being sent back by the server in every reply. Nothing
read them.

These checks cover the replacement:

- A scenario that asks for a **single word** back publishes no writing speed. Timing
  something that has barely begun is noise, and printing it next to a real figure invites
  it to be read as one.
- Three timed runs are combined with a **median, not an average**, so one unlucky run —
  a background process, the machine briefly busy — cannot drag the headline. The test uses
  a run that would move a mean from 50 to 34.
- A **warm-up run happens and is thrown away**, because the first attempt measures a cold
  start rather than the setup. One check proves it really ran; another proves it wasn't
  counted.
- A server that doesn't report its own timings gets **no figure at all** rather than the
  old blended one under a better name.
- A prompt bigger than the model's context is **skipped with a reason**, not attempted and
  scored zero.
- And the one that matters most: these speed figures are **kept out of the list of things
  that may be pooled across machines**. Speed is a fact about a computer. Pooling would
  average a laptop with a desktop and present the result as a property of the model.

### `test_aggregation.py` (9 checks) — a question never asked isn't a question failed

When a machine can't handle the biggest test, that test is **skipped**. A skip is neither
a pass nor a fail; it's an absence.

These prove a skipped item is left out of the average entirely, counted separately from
genuine errors, and never quietly read as a zero. There's also a check for the edge case
where a test module graded *nothing at all* — the counts must still make sense rather
than crashing or reporting a hopeful zero.

Two more guard the blended speed figure described above from coming back: it is easy to
re-add, it looks useful, and it is the wrong number.

### `test_metric_n.py` and `test_metric_counts.py` (3 + 3) — every number carries its receipts

The bundled question sets are small. **An accuracy of 0.83 over six questions is one
question away from 0.67.** Published bare, it reads as far more solid than it is.

So every figure is stored with `n` — how many graded items it rests on. These tests prove
the count is produced *by the same code that produces the figure*, in the same breath.
The rejected alternative — counting them up later with a separate database query — would
re-implement the filtering logic, and then the two could disagree without anyone noticing.

`test_metric_counts.py` covers the three trickiest cases, where "how many items" isn't
simply "all of them":

- Needle recall counts the cells probed *at that context length*, not overall.
- "Effective context" counts **rungs of the ladder**, not individual questions.
- Coding pass rate counts every attempt, including the failed ones.

### `test_unusable_response.py` (9 checks) — silence is not a wrong answer

**The bug, 2026-08-05 — the first time this bench met a real reasoning model.** The model
put its private thinking into a separate field, spent its entire 32-token answer budget
on that thinking, and returned an **empty** answer with the reason "ran out of room".

Every question scored 0.00. Read from the dashboard, that says *this model cannot
retrieve information* — a damning and completely false conclusion. The truth was *we
didn't give it enough room to speak*.

These tests use the real captured response and prove:

- A cut-off answer is flagged as cut off.
- A cut-off answer with no content **cannot be graded** — it's excluded, not zeroed.
- A model that finished properly and genuinely said nothing *is* still graded zero, which
  is correct and a different situation entirely.
- And a sweeping check: **every single grading module in the project** is walked through
  and confirmed to refuse to score a non-answer. Not just the one where the bug appeared.

### `test_shared_result_path.py` (18 checks) — the same scaffolding, written once

**The defect, found by audit rather than by failure.** Every test module used to copy the
same six measurements off the model's response onto its own result row — tokens in and
out, how long it took, how fast it ran, and the two speeds the server itself reports
(*reading the question* and *writing the answer*, which are different operations with
different bottlenecks).

Copied eleven times, the copies drifted. Two modules recorded all six. One recorded five.
**The other eight recorded four**, so the instruction-following and multiple-choice tests
reported no server-side speed at all. Nobody decided that. It is simply what copying
produces, and nothing failed, so nothing complained.

The scaffolding now lives in one place that fills every field, and these checks prove it:

- **Every module that asks a model anything records all six measurements** — walked
  through one module at a time, with a companion check that a module added later must be
  added to the list or this fails rather than quietly skipping the new one.
- A question file with a broken line reports **the file name and the line number**
  (blank lines don't shift the count), instead of the bare "expecting value" that four
  separate hand-written loaders used to produce for a file people edit by hand.
- Per-category figures — accuracy by subject, recall by context length — are now
  *declared* by a module rather than looped over by hand, and these prove the declaration
  produces the same figures the loops did: correct averages, each carrying its count, an
  unanswered question left out rather than averaged in as a zero, and an item with no
  category left out rather than filed under an invented one.

### `test_store_skipped.py` / `test_dashboard_skipped.py` (3 + 3) — a gap must explain itself

A skip is stored **with its reason**, and the dashboard leaves that point off the chart
rather than plotting a zero. A gap with no explanation reads as a score of zero, which is
the one thing this project never displays.

### `test_effort.py` (5 checks) — how much testing has this setup actually had

"Scored 0.83" means something different after 1 run than after 30. So each setup carries
a count of runs, total graded items, the date span, and which machines it ran on.

Crucially these figures are **counted fresh from the stored records every time** — nothing
is incremented, nothing is cached. So they can't be inflated and can't go stale. Failed
and partial runs are counted openly rather than hidden.

---

# Group 3 — "Can it start and stop model servers?"

Being able to launch the server is what makes the whole identity problem solvable: **a
bench that starts the server knows the settings, because it supplied them.**

### `test_launcher_profiles.py` (18 checks) — the list of things allowed to run

You describe the servers you want to be startable in a file (`~/.llmbench/servers.yaml`).
That file is also a **security allowlist**: the web dashboard may ask to start a profile
*by name*, and can never supply a program path or arguments. A web page that could post an
executable path plus arguments is a way to run anything on the machine hosting it.

Tests: profiles load correctly, arguments keep their order, a missing file means "you have
no profiles" rather than an error (most people never make one), a broken profile names
itself in the error, and — a nice detail — writing your arguments as one long string
instead of a list is **refused**, because splitting on spaces would mangle any path
containing a space, which on Windows is the normal case.

The rest cover **shared settings**. Anyone testing seriously ends up with dozens of these
profiles, almost identical, and copies that are almost identical drift apart. So a
`defaults` block holds the parts that repeat, and `{models}` style shortcuts stand in for
long folder paths:

- A profile's **own settings win** over the shared ones — "the usual, except this" only
  works if the exception is applied last. One check proves that a profile asking for a
  different graphics-card split really gets its own value.
- Shortcuts are **filled in when the file is read**, so what reaches the rest of the
  program is a real path and a finished command line, never a half-written one.
- The one that matters most: a profile that **inherits** a setting and one that
  **spells it out** are recognised as *the same setup*. Otherwise tidying up your own
  profile file would split a configuration's history in two and nothing would say why.
- A shortcut with **no value defined** is an error that names it. Left alone it would
  turn up later as a missing file, which blames the disk for a typing mistake.
- Braces that aren't a shortcut — a chat template, which is full of them — are **left
  untouched**.

### `test_launcher.py` (11 checks) — really starting, really stopping

These spawn a **real child process** on a **real port**. Rather than the real
`llama-server` (which needs a graphics card and a multi-gigabyte model — neither exists on
the free machines that run these checks), they launch a tiny stand-in that answers the
same way. It goes through exactly the same code path.

- Starting returns only once the server actually answers.
- A server that dies shows you **its own error output** — the useful part.
- A server that never answers is timed out *and cleaned up*.
- Stopping really ends the process (checked, not assumed). Stopping twice is harmless.
- If the code using the server crashes, the server is **still** stopped. An abandoned
  model server holds gigabytes of graphics memory and blocks the port the next run wants,
  and neither symptom points back at the bench that leaked it.

### `test_launcher_readiness.py` (4 checks) — "listening" is not "ready"

**The bug, 2026-08-05:** `llama-server` opens its network port *immediately*, long before
it has finished loading the model. Asked anything in that window it replies "service
unavailable". The old check treated *any* reply as "ready", so llmbench charged ahead and
tried to identify a server that couldn't yet describe itself. With an 11.8 GB model this
failed every single time.

Now: "still loading" is not ready; loaded is ready; and — the subtlety — a server that
replies "I don't know that request" **is** ready, because that means it's up and merely
dislikes that particular question. Nothing listening at all is not ready.

### `test_cli_launcher.py` (6) and `test_dashboard_servers.py` (9) — driving it

The same launch controls from the command line and from the web page. The dashboard tests
include the important negative ones:

- Starting an unknown profile is refused **and starts nothing**.
- The API offers **no way at all** to supply a program path — this is asserted directly,
  so nobody can add one by accident.
- Starting an already-running server doesn't start a second copy.
- A profile that fails returns the server's own error text to the page.

### `test_sweep.py` (5 checks) — testing several builds in a row

This is the workflow the project exists for: run the same tests against build A, then
build B, and compare. Tests prove each build is filed **separately**, no server is left
running afterwards, a typo in the fourth profile name fails **immediately** rather than
after three full test runs, and — importantly — **one build failing to start does not
abandon the builds after it**, because a pull request that doesn't compile is the normal
case when you're testing pull requests.

---

# Group 4 — "Do we know what machine this ran on?"

Speed depends entirely on the hardware. Quality does not. So quality figures from two
machines may be pooled; speed figures never may.

### `test_hostinfo.py` (4 checks) — facts about the computer

Notable for what it *doesn't* do. It asserts **properties** rather than values — "the
memory reading is plausible for any machine that could run this", not "this machine has 8
processors", which would only pass on the machine that wrote it. These checks run on three
different operating systems.

### `test_hostinfo_devices.py` (10 checks) — reading the graphics cards

No endpoint of the model server reports a single hardware fact (this was verified against
a real server, and written up). The graphics cards are read by asking the llama.cpp binary
to list them.

The test fixture is **real captured output**, confirmed byte-for-byte against a live run.
A hand-written one would only prove the parser matches its author's imagination.

The parser is deliberately ignorant: it reads **any** backend name without a built-in list
of them, so a graphics API that doesn't exist yet still parses. Also covered: a card whose
name contains brackets or digits, a machine with no cards (empty list, not an error), and
a binary that isn't there (unknown, not a crash).

### `test_hostinfo_drivers.py` (9 checks) — driver versions, recorded but not hashed

The graphics driver version is recorded but deliberately **excluded** from the machine's
identity. Reason: drivers update often, and hashing it would fragment a machine's history
every time. But it's *recorded*, so if a driver update is ever seen to move results,
there's a record to point at.

One test exists purely to assert that: **updating your driver does not turn your machine
into a different machine.**

### `test_host_identity.py` (10 checks) — what makes it "a different machine"

- Different graphics card → different machine. ✅
- Same card driven through a different compute backend → different machine. ✅
- **Free** memory (which fluctuates constantly) → same machine. ✅
- Total memory shifting by a few hundred megabytes between readings → same machine
  (there's a tolerance). A genuinely different amount of RAM → different machine.
- A minor OS point-release → same machine. A different OS → different machine.

### `test_store_host_grouping.py` (6 checks) — the rule that quality pools and speed doesn't

Every test here builds one setup measured on two machines with **identical quality and
different speed**, because that's the only shape where getting it wrong is visible: pool
the speeds and you produce the speed of a machine that never existed.

Also: an unfamiliar metric is treated as machine-dependent by default — the safe
assumption.

---

# Group 5 — "What will this cost in memory?"

### `test_gguf.py` (6 checks) — reading a model file's header

Model files carry a header describing the model's shape. The bench reads just that header,
which costs nothing — no model loading.

The tests **build tiny model files byte by byte**, because a real fixture was impossible
(the smallest model on the development machine was 83 MB). The parser was separately
validated against three real models, where what it read matched what a running server
independently reported.

Includes a nice detail: model headers contain the tokenizer's whole vocabulary — tens of
thousands of words this project never uses. The parser **steps over** it rather than
loading it, and the test proves it still reaches the settings that come *after* the
skipped block.

### `test_memory.py` (17 checks) — the arithmetic, "including the case where the obvious formula is 41× wrong"

That quote is the file's own opening line.

Some modern models only remember a sliding window of recent text rather than everything.
Applying the straightforward formula to such a model overstates its memory cost enormously.
There's a recorded lesson about this: an earlier estimate was wrong by **2.4×** and looked
completely plausible.

The proof these tests rest on is the strongest kind available: a whole block of them
**compares the calculation against what a real server actually allocated** — split cache,
unified cache, compressed, keys and values compressed differently, and the uncompressed
baseline for the same setup. Not "the formula matches itself"; "the formula matches
reality". One of these figures read 2.31 GiB until 2026-08-04, when a real server's own
allocation showed it was wrong.

And when the model's shape isn't recognised, the answer is **"unknown"** — never a guess
wearing the costume of a number.

### `test_cli_memory.py` (3 checks) — asking before you commit

You can ask what a setup *would* cost with **nothing running and no model loaded** — the
point being you can ask about a model you're merely considering. Compressing the cache
shows a smaller figure, and an unreadable model says "unknown" and exits with an error
rather than printing a comforting number.

---

# Group 6 — "Does the data survive?"

### `test_store_location.py` (2) — the database is always the same database

Its location doesn't depend on which folder you happened to be standing in when you ran
the command. Otherwise "where did my results go" becomes a regular question.

### `test_store_migration.py` (2) — old databases gain new columns

**The bug:** the code only ever said "create this table if it doesn't exist" — which does
nothing at all to a database that already exists. So any column added after the first
release would never reach anyone who'd already used the tool. Their results would just
stop recording the new fields.

The test starts from the **real historical database shape, frozen into the test file on
purpose**. Building the "old" database from the current definition would test nothing,
because that definition moves along with the code.

### `test_store_fingerprint_fields.py`, `test_store_host.py` (4 + 4) — what's hashed is also written down

If a setting is part of the identity, the setting itself must also be **stored** — or
later, when you see two setups that differ, nothing can tell you *how* they differ.
Also: the same machine seen twice is one record, not two.

An "unknown" memory estimate is stored as genuinely empty, **not as zero**. Zero is a
claim; empty is an absence.

---

# Group 7 — "Does what you see match what happened?"

### `test_dashboard_counts.py` (7 checks) — no naked figures on screen

Every figure served to the page carries its sample count. The heatmap says how many
samples are behind **each cell**, and a cell nobody probed counts **zero rather than
one** — the difference between "we tested this and found nothing" and "we never looked".

### `test_dashboard_configs.py` (3 checks) — showing the pooled figures

A whole phase of work built pooled quality and speed figures and **nothing ever displayed
them**. These make sure the page actually serves them.

### `test_cli_runs_table.py` (2 checks) — the output most likely to be pasted elsewhere

The `llmbench runs` table is what people screenshot into chat. So it carries the machine
name and the counts, and a run with no recorded machine **says so** rather than leaving a
suggestive blank.

### `test_cli_host.py` (5 checks) — showing the machine

Includes: a fact you declared by hand is visibly labelled **(declared)** rather than
passing itself off as something measured.

### `test_cli_encoding.py` (2 checks) — the terminal that can't print its own output

**The bug:** on Windows, output redirected to a file uses an older, narrower character
set. The formatting library silently substitutes a `?` for *some* characters it can't
represent — but not all. A particular line-drawing character in the sweep output raised an
error and **killed the entire run**.

That selectiveness is the trap. "It printed fine when I piped it" proves nothing about the
next character the tool learns to print.

These tests launch a **genuine separate process** with a legacy character set forced on,
because the fault is in how a program configures its own output when it starts — something
an in-process test replaces and therefore can never see.

### `test_docs_match_the_code.py` (3 checks) — the README can't drift

**The same defect shipped twice.** Both times a plan asked "does the new feature appear in
the README?", both times it did — in the shiny new section — and both times the
architecture map further down the same file still described an older program.

So instead of checking a list of edits (a check derived from the edit list is blind exactly
where the edit list is blind), this asserts a property of the whole file: **every module
and every command must appear in the map.** It fails when someone adds either and forgets
— which is the failure that actually happened, twice.

---

# Group 8 — "Does it survive the real world?"

### `test_ladder.py` (9 checks) — climbing until the machine says no

The bench tests progressively longer inputs — 8k, 16k, 32k... — until the machine can't
take any more. Two rules stop the climb: a rung that **fails**, and a rung projected to
**exceed the time budget**.

The clock is faked, so what would be a minutes-long wait runs in microseconds. Every test
asserts **which** rungs ran and **why** the rest didn't — because asserting merely that
"something was skipped" would pass on a climber that skipped everything.

### `test_graceful_degradation.py` (13 checks) — the machine hitting its limit is not a failure

The largest behavioural file, and it deliberately drives the **real** code against a
target that **really** refuses, rather than testing the two halves separately — because
testing either side of a join without testing the join is exactly how this project once
shipped a feature that could never fire.

- Rungs below the limit are **really graded** (not merely "nothing crashed").
- Rungs above are recorded as **skipped, with a reason**.
- One skipped row per rung, not one per question — otherwise the skips would swamp the
  results.
- A test module that crashes **does not cost the modules after it**, and is recorded as
  an error rather than vanishing.
- A target that can't be identified doesn't abandon the remaining targets.
- A module that crashes while *summarising* costs only itself.
- A machine at its limit still produces a **completed** run. This is the philosophical
  centre of the file: **a machine's honest limit is not a fault.**

### `test_coding_harness.py` (3 checks) — running code the model wrote

The coding test runs code an AI generated. Three checks: correct code passes, non-English
characters in the output don't break it, and — the important one — **a timeout kills the
whole process tree**, so nothing the generated code started outlives it.

Worth being clear, and the README is: this is a subprocess with a timeout, **not a
sandbox**. There are no memory or filesystem limits on any platform. Adversarial output
belongs in a container.

### `test_registry.py` (4 checks) — finding all the test modules

**The bug:** the discovery code treated "the list isn't empty" as proof that discovery had
already run. So if any code imported a single test module first, that one module made the
list look complete and the **other ten were never found**.

The first test is that exact scenario: import one, then check all eleven appear.

The other three cover test modules that arrive from **somebody else's package**. You can
extend this tool by installing an add-on rather than by forking it, so these build a real
add-on package in a temporary folder, hand it to a freshly started copy of the program,
and check what it makes of it:

- The add-on's test module shows up alongside the built-in ones.
- An add-on that **reuses a built-in name** is refused. Two things answering to the same
  name would mean your suite quietly ran the wrong one.
- An add-on that is **broken** stops the program with an error naming it, rather than
  being skipped. A test module that silently isn't there looks exactly like one you never
  installed — and then the bench runs fewer tests than you asked for without telling you.

They use a genuine package on disk rather than a stand-in, because the thing being tested
*is* Python's own add-on discovery, and a stand-in would sit on both sides of it without
ever touching it — a mistake this project has made before and written down.

### `test_resources.py` (4 checks) — bundled data is always found

The question sets ship inside the package. Before this, running from the wrong folder
would silently substitute toy data — you'd get results, they'd just be measuring the wrong
thing. Now bundled data is found from any folder, and a data file you configured but that
doesn't exist is a **hard error** rather than a silent fallback.

### `test_end_to_end.py` (1 check) — the whole thing, against a fake model

One test, the broadest in the suite. It runs **every test module** against a stand-in
model server that answers correctly wherever it can. That last part is what makes it
worth having: it proves the graders **accept correct answers**, rather than merely
proving they run.

---

## The three ideas behind all of it

If you remember nothing else:

**1. A number without its context is worse than no number.** Every figure carries how many
items it rests on, which machine produced it, and what settings were in force. A bare
0.83 is a claim nobody can check.

**2. Unknown is a valid answer; zero is not.** A skipped test, an unreadable model, a
server that wouldn't answer — all report *unknown*. The project's most-repeated failure
mode is a missing thing rendering as a zero, and a zero is a measurement, so it lies.

**3. Test the join, not just the parts.** Repeatedly, this project shipped features where
both halves were tested and the connection between them was broken. So the tests spawn
real processes, use real captured server responses, and compare the memory arithmetic
against what a real server really allocated.
