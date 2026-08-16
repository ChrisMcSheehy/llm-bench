"""Held-out tests for HumanEval/120, from OpenAI's HumanEval (MIT). See HUMANEVAL.md.

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
from solution import maximum

def check(candidate):

    # Check some simple cases
    assert candidate([-3, -4, 5], 3) == [-4, -3, 5]
    assert candidate([4, -4, 4], 2) == [4, 4]
    assert candidate([-3, 2, 1, 2, -1, -2, 1], 1) == [2]
    assert candidate([123, -123, 20, 0 , 1, 2, -3], 3) == [2, 20, 123]
    assert candidate([-123, 20, 0 , 1, 2, -3], 4) == [0, 1, 2, 20]
    assert candidate([5, 15, 0, 3, -13, -8, 0], 7) == [-13, -8, 0, 0, 3, 5, 15]
    assert candidate([-1, 0, 2, 5, 3, -10], 2) == [3, 5]
    assert candidate([1, 0, 5, -7], 1) == [5]
    assert candidate([4, -4], 2) == [-4, 4]
    assert candidate([-10, 10], 2) == [-10, 10]

    # Check some edge cases that are easy to work out by hand.
    assert candidate([1, 2, 3, -23, 243, -400, 0], 0) == []

def test_humaneval():
    check(maximum)
