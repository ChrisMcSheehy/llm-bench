from solution import rle_encode

def test_basic():
    assert rle_encode("aaabbc") == "a3b2c1"

def test_empty():
    assert rle_encode("") == ""

def test_single():
    assert rle_encode("x") == "x1"

def test_alternating():
    assert rle_encode("abab") == "a1b1a1b1"
