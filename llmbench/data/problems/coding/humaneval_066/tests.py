"""Held-out tests for HumanEval/66, from OpenAI's HumanEval (MIT). See HUMANEVAL.md.

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
from solution import digitSum

def check(candidate):

    # Check some simple cases
    assert True, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate("") == 0, "Error"
    assert candidate("abAB") == 131, "Error"
    assert candidate("abcCd") == 67, "Error"
    assert candidate("helloE") == 69, "Error"
    assert candidate("woArBld") == 131, "Error"
    assert candidate("aAaaaXa") == 153, "Error"

    # Check some edge cases that are easy to work out by hand.
    assert True, "This prints if this assert fails 2 (also good for debugging!)"
    assert candidate(" How are yOu?") == 151, "Error"
    assert candidate("You arE Very Smart") == 327, "Error"

def test_humaneval():
    check(digitSum)
