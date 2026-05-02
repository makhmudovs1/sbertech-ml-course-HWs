import numpy as np
from solution import StandardScaler
from common import assert_ndarray_equal

def test_standard_scaler_1():
    scaler = StandardScaler()

    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    scaler.fit(X)
    correct_mean = np.array([4.0, 5.0, 6.0])
    correct_var = np.array([6.0, 6.0, 6.0])

    assert_ndarray_equal(actual=scaler.mean_, correct=correct_mean,  err_msg="Test 1 Failed: incorrect mean")
    assert_ndarray_equal(actual=scaler.var_, correct=correct_var,  err_msg="Test 1 Failed: incorrect var")


def test_standard_scaler_2():
    scaler = StandardScaler()
    
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    correct_X_scaled = np.array([
        [-1.22474487, -1.22474487, -1.22474487],
        [0.        ,  0.        ,  0.        ],
        [1.22474487,  1.22474487,  1.22474487]])

    assert_ndarray_equal(actual=X_scaled, correct=correct_X_scaled,  err_msg="Test 2 Failed")

def test_standard_scaler_3():
    scaler = StandardScaler()
    
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    X_scaled = scaler.fit_transform(X)
    correct_X_scaled = np.array([
        [-1.22474487, -1.22474487, -1.22474487],
        [0.        ,  0.        ,  0.        ],
        [1.22474487,  1.22474487,  1.22474487]])
    
    assert_ndarray_equal(actual=X_scaled, correct=correct_X_scaled,  err_msg="Test 3 Failed")


def test_standard_scaler_4():
    scaler = StandardScaler()
    
    X = np.array([[1, 1],
                  [1, 1],
                  [1, 1]])
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    correct_X_scaled = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]])

    assert_ndarray_equal(actual=X_scaled, correct=correct_X_scaled,  err_msg="Test 4 Failed")

