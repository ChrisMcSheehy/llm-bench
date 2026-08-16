"""Held-out tests for HumanEval/144, from OpenAI's HumanEval (MIT). See HUMANEVAL.md.

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
from solution import simplify

def check(candidate):

    # Check some simple cases
    assert candidate("1/5", "5/1") == True, 'test1'
    assert candidate("1/6", "2/1") == False, 'test2'
    assert candidate("5/1", "3/1") == True, 'test3'
    assert candidate("7/10", "10/2") == False, 'test4'
    assert candidate("2/10", "50/10") == True, 'test5'
    assert candidate("7/2", "4/2") == True, 'test6'
    assert candidate("11/6", "6/1") == True, 'test7'
    assert candidate("2/3", "5/2") == False, 'test8'
    assert candidate("5/2", "3/5") == False, 'test9'
    assert candidate("2/4", "8/4") == True, 'test10'


    # Check some edge cases that are easy to work out by hand.
    assert candidate("2/4", "4/2") == True, 'test11'
    assert candidate("1/5", "5/1") == True, 'test12'
    assert candidate("1/5", "1/5") == False, 'test13'

def test_humaneval():
    check(simplify)
