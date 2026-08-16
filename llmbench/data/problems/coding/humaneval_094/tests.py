"""Held-out tests for HumanEval/94, from OpenAI's HumanEval (MIT). See HUMANEVAL.md.

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
from solution import skjkasdkd

def check(candidate):

    # Check some simple cases
    assert candidate([0,3,2,1,3,5,7,4,5,5,5,2,181,32,4,32,3,2,32,324,4,3]) == 10, "This prints if this assert fails 1 (good for debugging!)"

    # Check some edge cases that are easy to work out by hand.
    assert candidate([1,0,1,8,2,4597,2,1,3,40,1,2,1,2,4,2,5,1]) == 25, "This prints if this assert fails 2 (also good for debugging!)"

    # Check some edge cases that are easy to work out by hand.
    assert candidate([1,3,1,32,5107,34,83278,109,163,23,2323,32,30,1,9,3]) == 13, "This prints if this assert fails 3 (also good for debugging!)"

    # Check some edge cases that are easy to work out by hand.
    assert candidate([0,724,32,71,99,32,6,0,5,91,83,0,5,6]) == 11, "This prints if this assert fails 4 (also good for debugging!)"
    
    # Check some edge cases that are easy to work out by hand.
    assert candidate([0,81,12,3,1,21]) == 3, "This prints if this assert fails 5 (also good for debugging!)"

    # Check some edge cases that are easy to work out by hand.
    assert candidate([0,8,1,2,1,7]) == 7, "This prints if this assert fails 6 (also good for debugging!)"

    assert candidate([8191]) == 19, "This prints if this assert fails 7 (also good for debugging!)"
    assert candidate([8191, 123456, 127, 7]) == 19, "This prints if this assert fails 8 (also good for debugging!)"
    assert candidate([127, 97, 8192]) == 10, "This prints if this assert fails 9 (also good for debugging!)"

def test_humaneval():
    check(skjkasdkd)
