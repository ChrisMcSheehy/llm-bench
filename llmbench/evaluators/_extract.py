"""Shared answer-extraction / grading helpers.

Kept deliberately small and dependency-free. Every capability evaluator that
grades free-text output (MCQA, math, SQL, structured) leans on these so the
parsing rules are consistent and fixable in one place.
"""
from __future__ import annotations

import json
import re
import string
from typing import Any, Optional

# ---- multiple choice ------------------------------------------------------
_LETTER_PATTERNS = [
    re.compile(r"answer\s*(?:is|:)\s*\(?([A-Z])\)?", re.I),
    re.compile(r"\bfinal\s+answer\s*(?:is|:)?\s*\(?([A-Z])\)?", re.I),
    re.compile(r"\(([A-Z])\)"),
    re.compile(r"\b([A-Z])[.:)]"),
]


def extract_choice(text: str, n_options: int) -> Optional[str]:
    """Return a single option letter (A, B, ...) or None."""
    if not text:
        return None
    valid = set(string.ascii_uppercase[:n_options])
    for pat in _LETTER_PATTERNS:
        for m in reversed(list(pat.finditer(text))):
            if m.group(1).upper() in valid:
                return m.group(1).upper()
    # last resort: a lone valid letter anywhere, preferring the last
    for ch in reversed(text.strip()):
        if ch.upper() in valid:
            return ch.upper()
    return None


# ---- numeric (math) -------------------------------------------------------
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_HASH = re.compile(r"####\s*(.+)")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in (_BOXED, _HASH):
        m = pat.search(text)
        if m:
            nums = _NUM.findall(m.group(1))
            if nums:
                return _norm_num(nums[-1])
    nums = _NUM.findall(text)
    return _norm_num(nums[-1]) if nums else None


def _norm_num(s: str) -> str:
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def numbers_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


# ---- text normalisation (EM / F1) ----------------------------------------
def normalise(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalise(pred).split(), normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common: dict[str, int] = {}
    for t in p:
        if t in g:
            common[t] = common.get(t, 0) + 1
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


# ---- JSON extraction ------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.I)


def first_json(text: str) -> Optional[Any]:
    """Best-effort parse of the first JSON value in a model response."""
    if not text:
        return None
    candidates = _FENCE.findall(text) or [text]
    for cand in candidates:
        cand = cand.strip()
        # try whole, then the first {...} / [...] span
        for attempt in (cand, _bracket_span(cand)):
            if not attempt:
                continue
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    return None


def _bracket_span(s: str) -> Optional[str]:
    start = min([i for i in (s.find("{"), s.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    for i in range(start, len(s)):
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None
