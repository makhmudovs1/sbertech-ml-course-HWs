import numpy as np
from solution import one_hot_encode
from common import assert_ndarray_equal

def test_one_hot_encode_1():
    y = np.array([0, 1, 2])
    correct = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]], dtype=np.float64)
    actual = one_hot_encode(y)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 1 Failed")

def test_one_hot_encode_2():
    y = np.array([1, 2, 3])
    correct = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float64)
    actual = one_hot_encode(y)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 2 Failed")

def test_one_hot_encode_3():
    y = np.array([2, 2, 2, 2])
    correct = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]], dtype=np.float64)
    actual = one_hot_encode(y)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 3 Failed")

def test_one_hot_encode_4():
    y = np.array([0])
    correct = np.array([[1]], dtype=np.float64)
    actual = one_hot_encode(y)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 4 Failed")


def test_one_hot_encode_5():
    y = np.array([2, 1, 2, 0])
    correct = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0]], dtype=np.float64)
    actual = one_hot_encode(y, 4)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 5 Failed")