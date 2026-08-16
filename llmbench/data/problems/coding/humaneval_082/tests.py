"""Held-out tests for HumanEval/82, from OpenAI's HumanEval (MIT). See HUMANEVAL.md.

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
from solution import prime_length

def check(candidate):

    # Check some simple cases
    assert candidate('Hello') == True
    assert candidate('abcdcba') == True
    assert candidate('kittens') == True
    assert candidate('orange') == False
    assert candidate('wow') == True
    assert candidate('world') == True
    assert candidate('MadaM') == True
    assert candidate('Wow') == True
    assert candidate('') == False
    assert candidate('HI') == True
    assert candidate('go') == True
    assert candidate('gogo') == False
    assert candidate('aaaaaaaaaaaaaaa') == False

    # Check some edge cases that are easy to work out by hand.
    assert candidate('Madam') == True
    assert candidate('M') == False
    assert candidate('0') == False

def test_humaneval():
    check(prime_length)
