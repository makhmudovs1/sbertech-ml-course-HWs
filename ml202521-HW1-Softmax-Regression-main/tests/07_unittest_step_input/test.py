import numpy as np
from solution import SoftmaxRegression
from common import assert_ndarray_equal


def test_step_1():

    model = SoftmaxRegression(lr=0.1, penalty='l2', weight_decay=0, fit_intercept=True)
    model.coef_ = np.array([
        [8, 3, 2],
        [5, 6, 8],
        [1, 3, 0]], dtype=np.float64)
    model.intercept_ = np.array([7, 8, 5], dtype=np.float64)

    loss_grad = np.array([
        [-0.53333333,  0.16666667,  0.1       ],
        [ 0.43333333,  0.03333333, -0.4       ],
        [ 0.3       , -0.3       ,  0.33333333]])
    grad_intercept = np.array([ 0.31      , -0.27      ,  0.33333333])

    correct_coef = np.array([
        [ 8.053333,  2.983333,  1.99    ],
        [ 4.956667,  5.996667,  8.04    ],
        [ 0.97    ,  3.03    , -0.033333]])
    correct_intercept = np.array([6.969     , 8.027     , 4.96666667])

    model.step(loss_grad, grad_intercept)

    assert_ndarray_equal(correct=correct_coef, actual=model.coef_, err_msg='Test 1 Failed: incorrect coef')
    assert_ndarray_equal(correct=correct_intercept, actual=model.intercept_, err_msg='Test 1 Failed: incorrect intercept')

    
def test_step_2():

    model = SoftmaxRegression(lr=0.01, penalty='l2', weight_decay=0.1, fit_intercept=True)
    model.coef_ = np.array([
        [8, 3, 2],
        [5, 6, 8],
        [1, 3, 0]], dtype=np.float64)
    model.intercept_ = np.array([7, 8, 5], dtype=np.float64)
    
    loss_grad = np.array([
        [-0.45333333,  0.19666667,  0.12      ],
        [ 0.48333333,  0.09333333, -0.32      ],
        [ 0.31      , -0.27      ,  0.33333333]])
    grad_intercept = np.array([ 0.31      , -0.27      ,  0.33333333])

    correct_coef = np.array([
        [ 8.00453333e+00,  2.99803333e+00,  1.99880000e+00],
        [ 4.99516667e+00,  5.99906667e+00,  8.00320000e+00],
        [ 9.96900000e-01,  3.00270000e+00, -3.33333330e-03]])
    correct_intercept = np.array([6.9969    , 8.0027    , 4.99666667])

    model.step(loss_grad, grad_intercept)

    assert_ndarray_equal(correct=correct_coef, actual=model.coef_, err_msg='Test 2 Failed: incorrect coef')
    assert_ndarray_equal(correct=correct_intercept, actual=model.intercept_, err_msg='Test 2 Failed: incorrect intercept')


def test_step_3():

    model = SoftmaxRegression(lr=0.001, penalty='l1', weight_decay=0.1, fit_intercept=True)
    model.coef_ = np.array([
        [8, 3, 2],
        [5, 6, 8],
        [1, 3, 0]], dtype=np.float64)
    model.intercept_ = np.array([7, 8, 5], dtype=np.float64)
    
    loss_grad = np.array([
        [-0.45333333,  0.19666667,  0.12      ],
        [ 0.48333333,  0.09333333, -0.32      ],
        [ 0.31      , -0.27      ,  0.33333333]])
    grad_intercept = np.array([ 0.31      , -0.27      ,  0.33333333])

    correct_coef = np.array([
        [ 8.000453e+00,  2.999803e+00,  1.999880e+00],
        [ 4.999517e+00,  5.999907e+00,  8.000320e+00],
        [ 9.996900e-01,  3.000270e+00, -3.333333e-04]])
    correct_intercept = np.array([6.99969   , 8.00027   , 4.99966667])

    model.step(loss_grad, grad_intercept)

    assert_ndarray_equal(correct=correct_coef, actual=model.coef_, err_msg='Test 3 Failed: incorrect coef')
    assert_ndarray_equal(correct=correct_intercept, actual=model.intercept_, err_msg='Test 3 Failed: incorrect intercept')


def test_step_4():

    model = SoftmaxRegression(lr=0.1, penalty='l2', weight_decay=0, fit_intercept=False)
    model.coef_ = np.array([
        [8, 3, 2],
        [5, 6, 8],
        [1, 3, 0]], dtype=np.float64)
    model.intercept_ = np.array([0, 0, 0], dtype=np.float64)

    loss_grad = np.array([
        [-0.53333333,  0.16666667,  0.1       ],
        [ 0.43333333,  0.03333333, -0.4       ],
        [ 0.3       , -0.3       ,  0.33333333]])

    correct_coef = np.array([
        [ 8.053333,  2.983333,  1.99    ],
        [ 4.956667,  5.996667,  8.04    ],
        [ 0.97    ,  3.03    , -0.033333]])
    correct_intercept = model.intercept_

    model.step(loss_grad, None)

    assert_ndarray_equal(correct=correct_coef, actual=model.coef_, err_msg='Test 4 Failed: incorrect coef')
    assert_ndarray_equal(correct=correct_intercept, actual=model.intercept_, err_msg='Test 4 Failed: incorrect intercept')

    
def test_step_5():

    model = SoftmaxRegression(lr=0.1, penalty='l2', weight_decay=0.1, fit_intercept=False)
    model.coef_ = np.array([
        [8, 3, 2],
        [5, 6, 8],
        [1, 3, 0]], dtype=np.float64)
    model.intercept_ = np.array([0, 0, 0], dtype=np.float64)
    
    loss_grad = np.array([
        [-0.45333333,  0.19666667,  0.12      ],
        [ 0.48333333,  0.09333333, -0.32      ],
        [ 0.31      , -0.27      ,  0.33333333]])

    correct_coef = np.array([
        [ 8.045333,  2.980333,  1.988   ],
        [ 4.951667,  5.990667,  8.032   ],
        [ 0.969   ,  3.027   , -0.033333]])
    correct_intercept = model.intercept_

    model.step(loss_grad, None)

    assert_ndarray_equal(correct=correct_coef, actual=model.coef_, err_msg='Test 5 Failed: incorrect coef')
    assert_ndarray_equal(correct=correct_intercept, actual=model.intercept_, err_msg='Test 5 Failed: incorrect intercept')


def test_step_6():

    model = SoftmaxRegression(lr=0.1, penalty='l1', weight_decay=0.1, fit_intercept=False)
    model.coef_ = np.array([
        [8, 3, 2],
        [5, 6, 8],
        [1, 3, 0]], dtype=np.float64)
    model.intercept_ = np.array([0, 0, 0], dtype=np.float64)
    
    loss_grad = np.array([
        [-0.45333333,  0.19666667,  0.12      ],
        [ 0.48333333,  0.09333333, -0.32      ],
        [ 0.31      , -0.27      ,  0.33333333]])

    correct_coef = np.array([
        [ 8.045333,  2.980333,  1.988   ],
        [ 4.951667,  5.990667,  8.032   ],
        [ 0.969   ,  3.027   , -0.033333]])
    correct_intercept = model.intercept_

    model.step(loss_grad, None)

    assert_ndarray_equal(correct=correct_coef, actual=model.coef_, err_msg='Test 6 Failed: incorrect coef')
    assert_ndarray_equal(correct=correct_intercept, actual=model.intercept_, err_msg='Test 6 Failed: incorrect intercept')

    