"""Held-out tests for HumanEval/103, from OpenAI's HumanEval (MIT). See HUMANEVAL.md.

`check` below is HumanEval's own, unmodified. The wrapper at the bottom is this
project's: the harness runs pytest against a module called `solution`, so the
candidate is imported from there and handed to `check` exactly as HumanEval intends.

The wildcard import is not laziness. A handful of these problems state a helper in
the prompt - `poly`, `encode_cyclic`, `encode_shift` - and HumanEval's own `check`
calls that helper directly as well as the candidate. Importing only the entry point
left those three failing with a NameError raised by the test rather than by the
answer. The explicit import below it is kept because it is the contract with the
model, and it fails with a readable name when the model ignores it.
"""
from solution import *  # noqa: F401,F403 - helpers that HumanEval's own checks call
from solution import rounded_avg

def check(candidate):

    # Check some simple cases
    assert candidate(1, 5) == "0b11"
    assert candidate(7, 13) == "0b1010"
    assert candidate(964,977) == "0b1111001010"
    assert candidate(996,997) == "0b1111100100"
    assert candidate(560,851) == "0b1011000010"
    assert candidate(185,546) == "0b101101110"
    assert candidate(362,496) == "0b110101101"
    assert candidate(350,902) == "0b1001110010"
    assert candidate(197,233) == "0b11010111"


    # Check some edge cases that are easy to work out by hand.
    assert candidate(7, 5) == -1
    assert candidate(5, 1) == -1
    assert candidate(5, 5) == "0b101"

def test_humaneval():
    check(rounded_avg)
