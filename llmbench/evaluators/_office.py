"""A small simulated company, and the tools that read and change it (design B4).

Everything about this bench rests on results being comparable across runs, and a
tool benchmark that touches a real calendar, a real clock or a real network is
comparable with nothing, including itself an hour later. So the company is invented, the
tools are pure functions over a dictionary, and **the clock is frozen**: "tomorrow at
14:00 UTC" resolves to the same instant on every run, forever.

The frozen instant will eventually look absurd as it recedes into the past. Changing it
invalidates comparison with every earlier run, which is the worse cost, so it is a
constant in bundled data and moves only deliberately and with a note.

A helper module, like `_ladder.py` and `_sizing.py`: it registers nothing.
"""
from __future__ import annotations

import copy
from typing import Any

#: Monday 2 March 2026, 09:00 UTC. A Monday morning so that "tomorrow" and "this week"
#: both land inside the working week and no scenario depends on which day it is run.
FROZEN_NOW = "2026-03-02T09:00:00Z"
TOMORROW = "2026-03-03"

_EMPLOYEES = [
    {"id": "E-1001", "name": "Ada Whitfield", "email": "ada.whitfield@northgate.example",
     "team": "Platform", "location": "Leeds", "manager": "E-1004"},
    {"id": "E-1002", "name": "Ben Okoro", "email": "ben.okoro@northgate.example",
     "team": "Platform", "location": "Bristol", "manager": "E-1004"},
    {"id": "E-1003", "name": "Cara Lindqvist", "email": "cara.lindqvist@northgate.example",
     "team": "Data", "location": "Leeds", "manager": "E-1004"},
    {"id": "E-1004", "name": "Dan Merrick", "email": "dan.merrick@northgate.example",
     "team": "Engineering", "location": "Leeds", "manager": None},
]

_ROOMS = [
    {"id": "R-01", "name": "Kestrel", "seats": 4, "location": "Leeds"},
    {"id": "R-02", "name": "Harrier", "seats": 10, "location": "Leeds"},
    {"id": "R-03", "name": "Merlin", "seats": 6, "location": "Bristol"},
]

#: One room is already taken at the hour the scenarios ask about, so "check, then book"
#: has something to discover and a model that books without checking can be wrong.
_BOOKINGS = [
    {"room_id": "R-01", "date": TOMORROW, "time": "14:00", "who": "E-1003"},
]

_RATES = {("EUR", "GBP"): 0.85, ("GBP", "EUR"): 1.18,
          ("USD", "GBP"): 0.79, ("GBP", "USD"): 1.27}


def new_office() -> dict[str, Any]:
    """A fresh company. Deep-copied so one scenario's bookings cannot reach the next."""
    return {"employees": copy.deepcopy(_EMPLOYEES), "rooms": copy.deepcopy(_ROOMS),
            "bookings": copy.deepcopy(_BOOKINGS), "tickets": []}


# ---- the tools -------------------------------------------------------------------
#
# Each returns a plain dict, which is what gets handed back to the model as the tool's
# result. An unknown argument is an ordinary "not found" rather than an exception: a model
# guessing a name wrong is a scored behaviour, not a crash of the bench.

def find_employee(office, name: str = "", **_) -> dict:
    wanted = (name or "").strip().lower()
    for person in office["employees"]:
        if wanted and (wanted in person["name"].lower()
                       or wanted == person["id"].lower()):
            return {"found": True, **person}
    return {"found": False, "reason": f"no employee matching {name!r}"}


def list_team(office, team: str = "", **_) -> dict:
    members = [p for p in office["employees"]
               if p["team"].lower() == (team or "").strip().lower()]
    return {"team": team, "count": len(members),
            "members": [{"name": p["name"], "email": p["email"]} for p in members]}


def room_availability(office, date: str = "", time: str = "", location: str = "",
                      **_) -> dict:
    taken = {b["room_id"] for b in office["bookings"]
             if b["date"] == date and b["time"] == time}
    free = [r for r in office["rooms"]
            if r["id"] not in taken
            and (not location or r["location"].lower() == location.lower())]
    return {"date": date, "time": time,
            "available": [{"id": r["id"], "name": r["name"], "seats": r["seats"],
                           "location": r["location"]} for r in free]}


def book_room(office, room_id: str = "", date: str = "", time: str = "", **_) -> dict:
    room = next((r for r in office["rooms"] if r["id"].lower() == (room_id or "").lower()
                 or r["name"].lower() == (room_id or "").lower()), None)
    if room is None:
        return {"booked": False, "reason": f"no room {room_id!r}"}
    clash = any(b["room_id"] == room["id"] and b["date"] == date and b["time"] == time
                for b in office["bookings"])
    if clash:
        return {"booked": False, "reason": f"{room['name']} is already booked then"}
    office["bookings"].append({"room_id": room["id"], "date": date, "time": time,
                               "who": "requester"})
    return {"booked": True, "room": room["name"], "date": date, "time": time}


def create_ticket(office, title: str = "", priority: str = "normal", **_) -> dict:
    ticket = {"id": f"T-{2000 + len(office['tickets'])}", "title": title,
              "priority": priority}
    office["tickets"].append(ticket)
    return {"created": True, **ticket}


def exchange_rate(office, base: str = "", quote: str = "", **_) -> dict:
    rate = _RATES.get(((base or "").upper(), (quote or "").upper()))
    if rate is None:
        return {"known": False, "reason": f"no rate for {base}->{quote}"}
    return {"known": True, "base": base.upper(), "quote": quote.upper(), "rate": rate}


def convert_currency(office, amount: float = 0, base: str = "", quote: str = "",
                     **_) -> dict:
    rate = _RATES.get(((base or "").upper(), (quote or "").upper()))
    if rate is None:
        return {"converted": False, "reason": f"no rate for {base}->{quote}"}
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return {"converted": False, "reason": f"{amount!r} is not a number"}
    return {"converted": True, "amount": round(value * rate, 2), "currency": quote.upper()}


def archive_employee(office, employee_id: str = "", **_) -> dict:
    """The decoy. Plausible, in scope for a company, and never the right answer to
    anything asked. Reaching for it is what `focus` measures (design B4)."""
    office["employees"] = [p for p in office["employees"] if p["id"] != employee_id]
    return {"archived": True, "employee_id": employee_id}


#: name -> (function, description, argument names). The description is what the model is
#: shown; the argument names are only for the prompt.
TOOLS: dict[str, tuple] = {
    "find_employee": (find_employee, "Look up one employee by name or id.", ["name"]),
    "list_team": (list_team, "List the members of a named team.", ["team"]),
    "room_availability": (room_availability,
                          "Which meeting rooms are free at a date and time.",
                          ["date", "time", "location"]),
    "book_room": (book_room, "Book a meeting room at a date and time.",
                  ["room_id", "date", "time"]),
    "create_ticket": (create_ticket, "Raise a support ticket.", ["title", "priority"]),
    "exchange_rate": (exchange_rate, "The rate between two currency codes.",
                      ["base", "quote"]),
    "convert_currency": (convert_currency, "Convert an amount between two currencies.",
                         ["amount", "base", "quote"]),
    "archive_employee": (archive_employee, "Archive an employee record.",
                         ["employee_id"]),
}

#: Never the right answer to any scenario here. Named once so that the prompt, the
#: scoring and this module cannot disagree about which tool is the trap.
DECOY = "archive_employee"


def call(office: dict[str, Any], name: str, arguments: dict[str, Any]) -> dict:
    """Run one tool. An unknown name is a result, not an exception."""
    entry = TOOLS.get(name)
    if entry is None:
        return {"error": f"no tool named {name!r}"}
    if not isinstance(arguments, dict):
        return {"error": "arguments must be an object"}
    return entry[0](office, **arguments)
