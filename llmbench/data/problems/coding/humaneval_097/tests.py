"""Held-out tests for HumanEval/97, from OpenAI's HumanEval (MIT). See HUMANEVAL.md.

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
from solution import multiply

def check(candidate):

    # Check some simple cases
    assert candidate(148, 412) == 16, "First test error: " + str(candidate(148, 412))                    
    assert candidate(19, 28) == 72, "Second test error: " + str(candidate(19, 28))           
    assert candidate(2020, 1851) == 0, "Third test error: " + str(candidate(2020, 1851))
    assert candidate(14,-15) == 20, "Fourth test error: " + str(candidate(14,-15))      
    assert candidate(76, 67) == 42, "Fifth test error: " + str(candidate(76, 67))      
    assert candidate(17, 27) == 49, "Sixth test error: " + str(candidate(17, 27))      


    # Check some edge cases that are easy to work out by hand.
    assert candidate(0, 1) == 0, "1st edge test error: " + str(candidate(0, 1))
    assert candidate(0, 0) == 0, "2nd edge test error: " + str(candidate(0, 0))

def test_humaneval():
    check(multiply)
