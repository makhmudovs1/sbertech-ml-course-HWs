import numpy as np
from solution import StackingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

def test_binary():

    model = StackingClassifier()

    X = np.load(f'{__file__.replace("test.py", "")}/X_bin.npy')
    y = np.load(f'{__file__.replace("test.py", "")}/y_bin.npy')

    X, X_test, y, y_test = X[:20_000], X[20_000:], y[:20_000], y[20_000:]

    model.fit(X, y)

    assert len(model.estimators) >= 3, f'Мало базовых моделей: {len(model.estimators)}, должно быть не менее трех'
    assert model.final_estimator.n_features_in_ == len(model.estimators), f'Неверная входная размерность мета-модели. Должна быть {len(model.estimators)}'

    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    for estimator in model.estimators:
        s = roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1])
        assert s < score, f'У базовой модели {estimator} метрика ROC-AUC выше, чем у мета-модели, {s:.3f} > {score:.3f}'

    assert score >= 0.85, f'{score} < 0.85'


def test_multi():

    model = StackingClassifier()

    X = np.load(f'{__file__.replace("test.py", "")}/X_multi.npy')
    y = np.load(f'{__file__.replace("test.py", "")}/y_multi.npy')

    X, X_test, y, y_test = X[:9000], X[9000:], y[:9000], y[9000:]

    model.fit(X, y)

    assert len(model.estimators) >= 3, f'Мало базовых моделей: {len(model.estimators)}, должно быть не менее трех'
    assert model.final_estimator.n_features_in_ == len(model.estimators) * 9, f'Неверная входная размерность мета-модели. Должна быть {len(model.estimators) * 9}'

    score = accuracy_score(y_test, model.predict(X_test))

    for estimator in model.estimators:
        s = accuracy_score(y_test, estimator.predict(X_test))
        assert s < score, f'У базовой модели {estimator} метрика Accuracy выше, чем у мета-модели, {s:.3f} > {score:.3f}'

    assert score >= 0.75, f'{score} < 0.75'
    
