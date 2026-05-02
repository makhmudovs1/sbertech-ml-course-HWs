import numpy as np
from solution import SoftmaxRegression
from common import assert_ndarray_equal


def test_grad_1():

    model = SoftmaxRegression(weight_decay=0, fit_intercept=False)
    model.coef_ = np.array([[8, 3, 2],
           [5, 6, 8],
           [1, 3, 0]])
    model.intercept_ = np.array([0, 0, 0])
    model.n_classes_ = 3

    y_true = np.array([0, 1, 2])
    X = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    
    correct = np.array([
        [ 0.038653, -0.044391,  0.005738],
        [ 0.095405, -0.060461, -0.034944],
        [ 0.073609,  0.073633, -0.147242]])

    actual, actual_intercept= model.loss_grad(X, y_true)

    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 1 Failed: grad incorrect")
    assert actual_intercept is None, 'Test 1 Failed: grad_intercept incorrect"'

    
def test_grad_2():

    model = SoftmaxRegression(penalty='l2', weight_decay=0.01, fit_intercept=False)
    model.coef_ = np.array([[8, 3, 2],
           [5, 6, 8],
           [1, 3, 0]])
    model.intercept_ = np.array([0, 0, 0])
    model.n_classes_ = 3

    y_true = np.array([0, 1, 2])
    X = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    
    correct = np.array([
        [ 0.198653,  0.015609,  0.045738],
        [ 0.195405,  0.059539,  0.125056],
        [ 0.093609,  0.133633, -0.147242]])

    actual, actual_intercept= model.loss_grad(X, y_true)

    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 2 Failed: grad incorrect")
    assert actual_intercept is None, 'Test 2 Failed: grad_intercept incorrect"'


def test_grad_3():

    model = SoftmaxRegression(penalty='l1', weight_decay=0.01, fit_intercept=False)
    model.coef_ = np.array([[8, 3, 2],
           [5, 6, 8],
           [1, 3, 0]])
    model.intercept_ = np.array([0, 0, 0])
    model.n_classes_ = 3

    y_true = np.array([0, 1, 2])
    X = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    
    correct = np.array([
        [ 0.048653, -0.034391,  0.015738],
        [ 0.105405, -0.050461, -0.024944],
        [ 0.083609,  0.083633, -0.147242]])

    actual, actual_intercept= model.loss_grad(X, y_true)

    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 3 Failed: grad incorrect")
    assert actual_intercept is None, 'Test 3 Failed: grad_intercept incorrect"'


def test_grad_4():

    model = SoftmaxRegression(weight_decay=0, fit_intercept=True)
    model.coef_ = np.array([[8, 3, 2],
           [5, 6, 8],
           [1, 3, 0]])
    model.intercept_ = np.array([7, 8, 5])
    model.n_classes_ = 3

    y_true = np.array([0, 1, 2])
    X = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    
    correct = np.array([
        [ 0.016704,  0.012353, -0.029057],
        [ 0.069082,  0.024223, -0.093306],
        [ 0.045735,  0.149678, -0.195413]])

    correct_intercept = np.array([ 0.131522,  0.186254, -0.317776])

    actual, actual_intercept= model.loss_grad(X, y_true)

    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 4 Failed: grad incorrect")
    assert_ndarray_equal(actual=actual_intercept, correct=correct_intercept,  err_msg="Test 4 Failed: grad incorrect")

    
def test_grad_5():

    model = SoftmaxRegression(penalty='l2', weight_decay=0.01, fit_intercept=True)
    model.coef_ = np.array([[8, 3, 2],
           [5, 6, 8],
           [1, 3, 0]])
    model.intercept_ = np.array([7, 8, 5])
    model.n_classes_ = 3

    y_true = np.array([0, 1, 2])
    X = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    
    correct = np.array([
        [ 0.176704,  0.072353,  0.010943],
        [ 0.169082,  0.144223,  0.066694],
        [ 0.065735,  0.209678, -0.195413]])

    correct_intercept = np.array([ 0.131522,  0.186254, -0.317776])

    actual, actual_intercept= model.loss_grad(X, y_true)

    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 5 Failed: grad incorrect")
    assert_ndarray_equal(actual=actual_intercept, correct=correct_intercept,  err_msg="Test 5 Failed: grad incorrect")



def test_grad_6():

    model = SoftmaxRegression(penalty='l1', weight_decay=0.01, fit_intercept=True)
    model.coef_ = np.array([[8, 3, 2],
           [5, 6, 8],
           [1, 3, 0]])
    model.intercept_ = np.array([7, 8, 5])
    model.n_classes_ = 3

    y_true = np.array([0, 1, 2])
    X = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    
    correct = np.array([
        [ 0.026704,  0.022353, -0.019057],
        [ 0.079082,  0.034223, -0.083306],
        [ 0.055735,  0.159678, -0.195413]])

    correct_intercept = np.array([ 0.131522,  0.186254, -0.317776])

    actual, actual_intercept= model.loss_grad(X, y_true)

    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 6 Failed: grad incorrect")
    assert_ndarray_equal(actual=actual_intercept, correct=correct_intercept,  err_msg="Test 6 Failed: grad incorrect")
