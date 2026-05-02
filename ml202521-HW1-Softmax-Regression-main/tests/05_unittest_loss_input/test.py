import numpy as np
from solution import SoftmaxRegression
from common import assert_ndarray_equal


def test_loss_1():

    model = SoftmaxRegression(weight_decay=0)
    model.coef_ = np.array([[8, 3, 2],
           [5, 6, 8],
           [1, 3, 0]])
    model.intercept_ = np.array([7, 8, 5])

    y_true = np.array([0, 1, 2])
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    
    correct = 0.5202159160882228
    actual = model.loss(y_true, probs)

    assert np.isclose(actual, correct), "Test 1 Failed"
    
def test_loss_2():

    model = SoftmaxRegression(penalty='l2', weight_decay=0.01)
    model.coef_ = np.array([[8, 3, 2],
           [5, 6, 8],
           [1, 3, 0]])
    model.intercept_ = np.array([7, 8, 5])

    y_true = np.array([0, 1, 2])
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    
    correct = 2.640215916088223
    actual = model.loss(y_true, probs)

    assert np.isclose(actual, correct), "Test 2 Failed"


def test_loss_3():

    model = SoftmaxRegression(penalty='l1', weight_decay=0.01)
    model.coef_ = np.array([[8, 3, 2],
           [5, 6, 8],
           [1, 3, 0]])
    model.intercept_ = np.array([7, 8, 5])

    y_true = np.array([0, 1, 2])
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    
    correct = 0.8802159160882228
    actual = model.loss(y_true, probs)

    assert np.isclose(actual, correct), "Test 3 Failed"

    