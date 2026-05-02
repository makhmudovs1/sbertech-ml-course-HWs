import numpy as np
from solution import SoftmaxRegression
from common import assert_ndarray_equal


def test_predict_1():

    model = SoftmaxRegression()
    model.coef_ = np.array([[8, 5, 1]])
    model.intercept_ = np.array([7, 2, 1])

    X = np.array([[-1.34783067],
       [-1.82698213],
       [-1.0395786 ],
       [-0.48313131],
       [ 0.54284994],
       [-2.31929004],
       [-0.19568926],
       [-2.4347564 ],
       [-0.76744448],
       [-2.9446212 ]])

    actual = model.predict(X)

    correct = np.array([2, 2, 2, 0, 0, 2, 0, 2, 0, 2])

    assert_ndarray_equal(correct=correct, actual=actual, err_msg='Test 1 Failed')

    
def test_predict_2():

    model = SoftmaxRegression()
    model.coef_ = np.array([
        [8, 3, 2],
        [5, 6, 8],
        [1, 3, 0], 
        [-1, 0, 9]])
    model.intercept_ = np.array([7, 8, 5])
    
    X = np.array([[-1.88410657, -1.4562849 , -0.26786086, -1.98337983],
       [ 0.49296843, -1.31316104, -1.33452649, -1.12234877],
       [ 0.17948176, -1.19334198, -1.02785634, -2.0610986 ],
       [-0.42685327, -1.73920213, -1.04990344, -3.48129608],
       [-1.04109474,  0.0756424 , -0.13264918, -1.58288797],
       [-2.3213775 , -3.44956942, -1.80228679,  0.28671583],
       [-1.83510376, -0.56379521, -2.11184355, -0.99271932],
       [-1.81245323, -0.29125698, -2.23863083, -2.37841275],
       [-0.64343871, -1.90509213, -0.55792398,  0.14948942],
       [ 0.68655732, -2.49724917, -1.50467435, -1.32191236]])
    
    actual = model.predict(X)

    correct = np.array([1, 0, 0, 0, 1, 2, 1, 1, 1, 0])

    assert_ndarray_equal(correct=correct, actual=actual, err_msg='Test 2 Failed')


def test_predict_3():

    model = SoftmaxRegression()
    model.coef_ = np.array([
        [8, 3, 2],
        [5, 6, 8]])
    model.intercept_ = np.array([7, 8, 5])
    
    X = np.array([[-1.02374812, -1.15634698],
       [-1.75073572, -0.87420576],
       [-2.27486   , -1.34683543],
       [-1.17265173,  0.68764482],
       [-1.15185405,  0.80976883],
       [-0.28348912, -2.84775159],
       [-0.72443317, -1.67208178],
       [-0.40572303, -1.67060923],
       [-0.99840023,  0.31546582],
       [-1.62097939, -2.04460417]])

    actual = model.predict(X)

    correct = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1])

    assert_ndarray_equal(correct=correct, actual=actual, err_msg='Test 3 Failed')

    