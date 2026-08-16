"""Held-out tests for HumanEval/89, from OpenAI's HumanEval (MIT). See HUMANEVAL.md.

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
from solution import encrypt

def check(candidate):

    # Check some simple cases
    assert candidate('hi') == 'lm', "This prints if this assert fails 1 (good for debugging!)"
    assert candidate('asdfghjkl') == 'ewhjklnop', "This prints if this assert fails 1 (good for debugging!)"
    assert candidate('gf') == 'kj', "This prints if this assert fails 1 (good for debugging!)"
    assert candidate('et') == 'ix', "This prints if this assert fails 1 (good for debugging!)"

    assert candidate('faewfawefaewg')=='jeiajeaijeiak', "This prints if this assert fails 1 (good for debugging!)"
    assert candidate('hellomyfriend')=='lippsqcjvmirh', "This prints if this assert fails 2 (good for debugging!)"
    assert candidate('dxzdlmnilfuhmilufhlihufnmlimnufhlimnufhfucufh')=='hbdhpqrmpjylqmpyjlpmlyjrqpmqryjlpmqryjljygyjl', "This prints if this assert fails 3 (good for debugging!)"

    # Check some edge cases that are easy to work out by hand.
    assert candidate('a')=='e', "This prints if this assert fails 2 (also good for debugging!)"

def test_humaneval():
    check(encrypt)
