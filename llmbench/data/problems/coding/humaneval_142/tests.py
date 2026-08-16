"""Held-out tests for HumanEval/142, from OpenAI's HumanEval (MIT). See HUMANEVAL.md.

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
from solution import sum_squares

def check(candidate):

    # Check some simple cases
    
    assert candidate([1,2,3]) == 6
    assert candidate([1,4,9]) == 14
    assert candidate([]) == 0
    assert candidate([1,1,1,1,1,1,1,1,1]) == 9
    assert candidate([-1,-1,-1,-1,-1,-1,-1,-1,-1]) == -3
    assert candidate([0]) == 0
    assert candidate([-1,-5,2,-1,-5]) == -126
    assert candidate([-56,-99,1,0,-2]) == 3030
    assert candidate([-1,0,0,0,0,0,0,0,-1]) == 0
    assert candidate([-16, -9, -2, 36, 36, 26, -20, 25, -40, 20, -4, 12, -26, 35, 37]) == -14196
    assert candidate([-1, -3, 17, -1, -15, 13, -1, 14, -14, -12, -5, 14, -14, 6, 13, 11, 16, 16, 4, 10]) == -1448
    
    
    # Don't remove this line:

def test_humaneval():
    check(sum_squares)
