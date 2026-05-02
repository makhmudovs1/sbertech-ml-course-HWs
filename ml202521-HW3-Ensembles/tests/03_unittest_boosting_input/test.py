import numpy as np
from solution import BoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


def test_binary():

    model = BoostingClassifier()

    X = np.load(f'{__file__.replace("test.py", "")}/X_bin.npy')
    y = np.load(f'{__file__.replace("test.py", "")}/y_bin.npy')

    X, X_test, y, y_test = X[:20_000], X[20_000:], y[:20_000], y[20_000:]

    model.fit(X, y)

    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    assert score >= 0.85, f'{score} < 0.85'


def test_multi():

    model = BoostingClassifier()

    X = np.load(f'{__file__.replace("test.py", "")}/X_multi.npy')
    y = np.load(f'{__file__.replace("test.py", "")}/y_multi.npy')

    X, X_test, y, y_test = X[:9000], X[9000:], y[:9000], y[9000:]

    model.fit(X, y)

    score = accuracy_score(y_test, model.predict(X_test))

    assert score >= 0.75, f'{score} < 0.75'
    