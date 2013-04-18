
import numpy as np
from ..skurious.rank_metrics import delta_avg, delta_avg_n, spearman

REFS = np.array([6,4,2,1,3,5])
REFS2 = np.array([6,4,2,1,3,5,7])


def test_delta_avg_1():
    expected = 0.5625
    preds = np.array([6,5,4,3,2,1])
    result = delta_avg(preds, REFS)
    assert result == expected


def test_delta_avg_2():
    expected = 0.0625
    preds = np.array([7,6,5,4,3,2,1])
    result = delta_avg(preds, REFS2)
    assert result == expected


def test_delta_avg_3():
    expected = 2
    preds = REFS2
    result = delta_avg(preds, REFS2)
    print result
    assert result == expected


def test_delta_avg_n2():
    expected = 0.5
    rank = REFS
    result = delta_avg_n(rank, 2, np.mean(rank))
    assert result == expected


def test_delta_avg_n3():
    expected = 0.625
    rank = REFS
    result = delta_avg_n(rank, 3, np.mean(rank))
    print result
    assert result == expected


def test_delta_avg_n22():
    expected = 0
    rank = REFS2
    result = delta_avg_n(rank, 2, np.mean(rank))
    assert result == expected


def test_delta_avg_n32():
    expected = 0.125
    rank = REFS2
    result = delta_avg_n(rank, 3, np.mean(rank))
    print result
    assert result == expected


def test_spearman_1():
    expected = -1
    rank = range(12,0,-2)
    ref = range(1,7)
    result = spearman(rank, ref)
    assert result == expected


def test_spearman_2():
    expected = 1
    rank = range(18,0,-3)
    ref = rank
    result = spearman(rank, ref)
    assert result == expected


def test_spearman_3():
    expected = 0.5
    rank = [40, 40, 40, 40, 40]
    ref = range(0, 10, 2)
    result = spearman(rank, ref)
    assert result == expected


def test_spearman_4():
    expected = 0.75
    rank = [10, 40, 40, 40, 40]
    ref = range(0, 10, 2)
    result = spearman(rank, ref)
    print result
    assert result == expected
