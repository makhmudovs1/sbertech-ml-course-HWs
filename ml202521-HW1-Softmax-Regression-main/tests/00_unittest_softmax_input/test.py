import numpy as np
from solution import softmax
from common import assert_ndarray_equal

def test_softmax_1():
  
    x = np.array([[1, 2, 3]])
    correct = np.array([[0.09003057, 0.24472847, 0.66524096]])
    actual = softmax(x)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 1 Failed")

def test_softmax_2():
    x = np.array([[1, 0.5, 0]])
    correct = np.array([[0.50648039, 0.30719589, 0.18632372]])
    actual = softmax(x)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 2 Failed")

def test_softmax_3():
    x = np.array([[1000, 1000, 1000]])
    correct = np.array([[0.33333333, 0.33333333, 0.33333333]])
    actual = softmax(x)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 3 Failed")

def test_softmax_4():
    x = np.array([[1000, 10000, 100000]])
    correct = np.array([[0., 0., 1.]])
    actual = softmax(x)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 4 Failed")

def test_softmax_5():
    x = np.array([[1, 2, 3], [3, 2, 1]])
    correct = np.array([[0.09003057, 0.24472847, 0.66524096],
       [0.66524096, 0.24472847, 0.09003057]])
    actual = softmax(x)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 5 Failed")

def test_softmax_6():
    x = np.array([[1, 2, 3], [6, 5, 4]])
    correct = np.array([[0.09003057, 0.24472847, 0.66524096],
       [0.66524096, 0.24472847, 0.09003057]])
    actual = softmax(x)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 6 Failed")

def test_softmax_7():
    x = np.array([[-1, -2, -3]])
    correct = np.array([[0.66524096, 0.24472847, 0.09003057]])
    actual = softmax(x)
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 7 Failed")
