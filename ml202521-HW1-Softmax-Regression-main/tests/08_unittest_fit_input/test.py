import numpy as np
from solution import SoftmaxRegression
from common import assert_ndarray_equal


def test_fit_1():

    model = SoftmaxRegression(lr=0.1, n_epochs=200)

    X = np.load(f'{__file__.replace("test.py", "")}/X.npy')
    y = np.load(f'{__file__.replace("test.py", "")}/y.npy')

    X, X_test, y, y_test = X[:100], X[100:], y[:100], y[100:]

    model.fit(X, y)

    acc = np.mean(model.predict(X_test) == y_test)

    assert acc >= 0.95, f'{acc} < 0.95'
    