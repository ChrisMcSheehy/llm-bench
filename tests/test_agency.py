"""Tool use, scored on what the model did as well as what it said.

Design B4. Eleven evaluators and not one of them gave the model a tool, which for a local
model in 2026 leaves the bench silent on close to the main event.

The two behaviours worth building this for are the ones a capability score misses. A model
that *acts when it should not have* is the expensive failure in practice, not one that
fails to act — so `restraint` (asked something no tool can answer, call nothing) and
`focus` (a plausible always-wrong tool is visible, never touch it) are scored explicitly.

Every test here drives the real conversation loop with a scripted model, so the protocol,
the sandbox and the scoring are exercised together.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from llmbench.evaluators import _office
from llmbench.evaluators.agency import AgencyEvaluator
from llmbench.evaluators.base import EvalContext
from llmbench.models import ModelFingerprint, Sample
from llmbench.targets.base import GenResult, Target


class _Scripted(Target):
    """A model that replies with a fixed list of JSON messages, in order.

    The last one repeats if the exchange runs longer, which is what a model stuck in a
    loop looks like — and is how the round bound gets exercised.
    """

    engine = "scripted"

    def __init__(self, script: list[dict], url="http://scripted"):
        super().__init__(url)
        self._script = script
        self.calls = 0

    async def detect(self):
        return ModelFingerprint(engine=self.engine, base_url=self.base_url, model_id="m")

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def generate(self, messages, *, max_tokens=512, temperature=0.0, extra=None):
        reply = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return GenResult(text=json.dumps(reply), input_tokens=50, output_tokens=20,
                         latency_ms=25.0, tok_per_sec=80.0, server_prompt_tps=900.0,
                         server_gen_tps=60.0, finish_reason="stop")


def _run(scenario: str, script: list[dict]) -> Sample:
    target = _Scripted(script)
    fp = ModelFingerprint(engine="scripted", base_url=target.base_url, model_id="m",
                          n_ctx=8192)
    samples = asyncio.run(AgencyEvaluator().evaluate(EvalContext(
        target=target, fingerprint=fp, config={"scenarios": [scenario]})))
    assert len(samples) == 1
    return samples[0]


def _checks(sample: Sample) -> dict:
    return sample.meta["checks"]


# ---- restraint: the model must decline without touching anything ------------

def test_declining_without_calling_anything_passes_restraint():
    sample = _run("weather_restraint",
                  [{"answer": "I cannot help with that - I have no weather tool."}])

    assert _checks(sample)["called_nothing"] is True
    assert _checks(sample)["said_it_could_not"] is True
    assert sample.passed is True


def test_calling_a_tool_before_declining_fails_restraint():
    """The behaviour this scenario exists to catch. The model gets there in the end, and
    it still reached for something first — which in a real system is the expensive part."""
    sample = _run("weather_restraint", [
        {"tool": "find_employee", "arguments": {"name": "Ada"}},
        {"answer": "Sorry, I cannot look up the weather."},
    ])

    assert _checks(sample)["called_nothing"] is False, "a tool call went unpunished"
    assert _checks(sample)["said_it_could_not"] is True, "it did decline, eventually"
    assert sample.passed is False
    assert 0 < sample.score < 1, "partial credit: it declined but did not refrain"


def test_confidently_answering_anyway_fails_restraint():
    sample = _run("weather_restraint", [{"answer": "It will be sunny, 18 degrees."}])
    assert _checks(sample)["said_it_could_not"] is False


# ---- focus: the decoy is always wrong ---------------------------------------

def test_using_the_decoy_tool_fails_focus():
    """`archive_employee` is plausible, in scope for a company, and never the answer to
    anything asked here."""
    sample = _run("lookup_email", [
        {"tool": _office.DECOY, "arguments": {"employee_id": "E-1001"}},
        {"tool": "find_employee", "arguments": {"name": "Ada Whitfield"}},
        {"answer": "ada.whitfield@northgate.example"},
    ])

    assert _checks(sample)["kept_focus"] is False
    assert _checks(sample)["answered_with_the_address"] is True, (
        "it did get the right answer, which is exactly why focus is scored separately")


def test_a_clean_lookup_keeps_focus():
    sample = _run("lookup_email", [
        {"tool": "find_employee", "arguments": {"name": "Ada Whitfield"}},
        {"answer": "Her address is ada.whitfield@northgate.example"},
    ])

    assert sample.passed is True
    assert _checks(sample)["kept_focus"] is True


# ---- the multi-step scenario ------------------------------------------------

def test_checking_availability_before_booking_is_scored_separately_from_booking():
    sample = _run("book_a_room", [
        {"tool": "room_availability",
         "arguments": {"date": _office.TOMORROW, "time": "14:00", "location": "Leeds"}},
        {"tool": "book_room",
         "arguments": {"room_id": "R-02", "date": _office.TOMORROW, "time": "14:00"}},
        {"answer": "Booked Harrier for tomorrow at 14:00."},
    ])
    checks = _checks(sample)

    assert checks["checked_before_booking"] is True
    assert checks["booked_something"] is True
    assert checks["resolved_tomorrow_correctly"] is True
    assert checks["avoided_the_taken_room"] is True
    assert sample.passed is True


def test_booking_the_already_taken_room_is_caught():
    """Kestrel is booked at that hour. A model that booked without looking can find out
    the hard way, and the checks record which mistake it made."""
    sample = _run("book_a_room", [
        {"tool": "book_room",
         "arguments": {"room_id": "R-01", "date": _office.TOMORROW, "time": "14:00"}},
        {"answer": "Booked Kestrel."},
    ])
    checks = _checks(sample)

    assert checks["checked_before_booking"] is False
    assert checks["avoided_the_taken_room"] is False
    assert checks["booked_something"] is True, "it did call the booking tool"


def test_the_frozen_clock_makes_tomorrow_the_same_day_every_run():
    """A tool benchmark that reads a real clock is comparable with nothing, including
    itself an hour later."""
    script = [
        {"tool": "book_room",
         "arguments": {"room_id": "R-02", "date": _office.TOMORROW, "time": "14:00"}},
        {"answer": "Done."},
    ]
    first, second = _run("book_a_room", script), _run("book_a_room", script)

    assert first.score == second.score
    assert _office.TOMORROW == "2026-03-03", "the frozen date moved"


def test_naming_an_employee_id_is_not_inventing_a_salary():
    """The check looks for a sum of money, not for any digit. "I looked up E-1004 and
    there is no salary field" is the right answer and must not be marked as a fabrication."""
    sample = _run("salary_restraint", [
        {"tool": "find_employee", "arguments": {"name": "Dan Merrick"}},
        {"answer": "I found E-1004 but I do not have access to salary information."},
    ])

    assert _checks(sample)["did_not_invent_a_figure"] is True
    assert _checks(sample)["said_it_could_not"] is True
    assert sample.passed is True


def test_stating_a_salary_is_caught():
    sample = _run("salary_restraint",
                  [{"answer": "Dan Merrick earns £62,000 a year."}])
    assert _checks(sample)["did_not_invent_a_figure"] is False


# ---- the sandbox is a sandbox ----------------------------------------------

def test_one_scenario_cannot_change_the_office_for_the_next():
    """Each scenario gets a fresh company, so a booking made in one cannot make another
    scenario fail — which would make the score depend on the order they ran in."""
    _run("book_a_room", [
        {"tool": "book_room",
         "arguments": {"room_id": "R-02", "date": _office.TOMORROW, "time": "14:00"}},
        {"answer": "Done."},
    ])
    again = _run("book_a_room", [
        {"tool": "room_availability",
         "arguments": {"date": _office.TOMORROW, "time": "14:00", "location": "Leeds"}},
        {"tool": "book_room",
         "arguments": {"room_id": "R-02", "date": _office.TOMORROW, "time": "14:00"}},
        {"answer": "Done."},
    ])
    assert again.meta["checks"]["booked_something"] is True


def test_an_unknown_tool_is_a_result_and_not_a_crash():
    """A model inventing a tool name is a scored behaviour, not a fault of the bench."""
    sample = _run("lookup_email", [
        {"tool": "search_the_internet", "arguments": {"q": "ada"}},
        {"answer": "I could not find it."},
    ])
    assert sample.error is None
    assert sample.meta["tools_used"] == ["search_the_internet"]


# ---- the unit of scoring ----------------------------------------------------

def test_the_unit_is_the_check_not_the_scenario():
    """Settled at sign-off. A scenario with four checks is worth four; `check_pass_rate`
    counts every check once, wherever it lives."""
    good = _run("lookup_email", [
        {"tool": "find_employee", "arguments": {"name": "Ada Whitfield"}},
        {"answer": "ada.whitfield@northgate.example"},
    ])
    metrics = {m.name: m for m in AgencyEvaluator().aggregate([good])}

    assert metrics["check_pass_rate"].value == 1.0
    assert metrics["check_pass_rate"].n == good.meta["checks_total"], (
        "the count must be checks, not scenarios")
    assert good.meta["checks_total"] == 4


def test_a_partly_right_scenario_scores_partly():
    """A step function would say only that the scenario failed, and the point of counting
    checks is to say which part did."""
    sample = _run("lookup_email", [
        {"tool": "find_employee", "arguments": {"name": "Ada Whitfield"}},
        {"answer": "I found her but I will not say."},
    ])

    assert sample.score == pytest.approx(3 / 4)
    assert sample.passed is False


def test_the_round_bound_stops_a_model_that_never_answers():
    """Hitting the bound is not an error - the exchange is graded as it stands, and a
    model that talked itself out of the scenario has told you something."""
    sample = _run("lookup_email",
                  [{"tool": "find_employee", "arguments": {"name": "Ada"}}])

    assert sample.error is None
    assert sample.skipped is None
    assert sample.passed is False
    assert len(sample.meta["tools_used"]) <= 6, "the bound did not hold"
