from solution import two_sum

def test_basic():
    assert two_sum([2, 7, 11, 15], 9) == (0, 1)

def test_middle():
    assert two_sum([3, 2, 4], 6) == (1, 2)

def test_negatives():
    i, j = two_sum([-3, 4, 3, 90], 0)
    assert i < j and {-3, 3} == {[-3,4,3,90][i], [-3,4,3,90][j]}
