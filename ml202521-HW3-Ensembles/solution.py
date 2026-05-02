import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

class RandomForestClassifier:
    """
    Модель случайного леса (RandomForest).

    Атрибуты:
    ----------
    n_estimators : int
        Количество деревьев в лесу.
    
    bootstrap : bool
        Используется ли бутстрап при построении деревьев. 
        Если False, то для каждого дерева используется весь набор данных.
    
    estimators : list
        Список деревьев с заданными параметрами (**kwargs).
    
    kwargs : dict
        Параметры для каждого дерева, такие как min_samples_split, max_depth и др.
        Передаются в DecisionTreeClassifier.
    """
    def __init__(self, n_estimators=100, bootstrap=True, kwargs=None):
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.kwargs = {} if kwargs is None else dict(kwargs)
        self.estimators = []
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Обучает случайный лес на тренировочной выборке (X, y). 
        Если bootstrap=False, используется весь набор данных для каждого дерева,
        если bootstrap=True, используются бутстрап-выборки.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Тренировочные данные (признаки).
        
        y : np.array, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : RandomForestClassifier
            Обученная модель.
        """

        n_samples = X.shape[0]
        self.estimators = []
        for i in range(self.n_estimators):
            tree_params = dict(self.kwargs)
            if "random_state" not in tree_params:
                tree_params["random_state"] = i
            tree = DecisionTreeClassifier(**tree_params)
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)

            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(tree)

        if self.estimators:
            self.classes_ = self.estimators[0].classes_
        return self

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов для каждого объекта на основе 
        предсказаний всех деревьев в лесу. Возвращает средние вероятности по всем деревьям.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания вероятностей классов.

        Возвращает:
        ----------
        np.array, shape (n_samples, n_classes)
            Вероятности для каждого класса и каждого объекта.
        """
        if not self.estimators:
            raise ValueError("Модель RandomForestClassifier еще не обучена (estimators пуст).")

        probs = []
        for tree in self.estimators:
            probs.append(tree.predict_proba(X))

        probs = np.stack(probs, axis=0)
        mean_probs = probs.mean(axis=0)
        return mean_probs
        
    def predict(self, X):
        """
        Предсказывает метки классов для каждого объекта входной выборки 
        на основе голосования всех деревьев.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов.
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]


class BlendingClassifier:
    """
    Модель ансамбля методом блендинга (Blending).

    Атрибуты:
    ----------
    estimators : list
        Список инициализированных базовых моделей.

    final_estimator : объект модели
        Метамодель, обучаемая на предсказаниях базовых моделей.

    test_size : float
        Доля данных, используемая для обучения метамодели (блендинга).

    """

    def __init__(self, estimators=None, final_estimator=None, test_size=0.3):
        if estimators is None:
            self.estimators = [
                LogisticRegression(
                    max_iter=1000,
                    multi_class="auto"
                ),
                SklearnRandomForestClassifier(
                    n_estimators=150,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=0
                ),
                KNeighborsClassifier(
                    n_neighbors=7
                ),
            ]
        else:
            self.estimators = estimators
        if final_estimator is None:
            self.final_estimator = LogisticRegression(
                max_iter=1000,
                multi_class="auto"
            )
        else:
            self.final_estimator = final_estimator

        self.test_size = float(test_size)
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        """
        Разделяет входную выборку на тренировочную и валидационную части.
        Базовые модели обучаются на тренировочной части, а метамодель — на предсказаниях
        базовых моделей на валидационной части.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Тренировочные данные (признаки).

        y : np.array, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : Blending
            Обученная модель.
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=42,
        )

        for est in self.estimators:
            est.fit(X_train, y_train)

        meta_features = []
        for est in self.estimators:
            proba = est.predict_proba(X_val)
            if self.n_classes_ == 2:
                meta_features.append(proba[:, 1:2])
            else:
                k = min(9, proba.shape[1])
                meta_features.append(proba[:, :k])

        meta_X = np.hstack(meta_features)

        self.final_estimator.fit(meta_X, y_val)
        self.classes_ = self.final_estimator.classes_
        return self

    def _build_meta_features(self, X):
        """
        Вспомогательный метод: сделать мета-признаки для произвольного X
        из predict_proba базовых моделей.
        """
        if self.n_classes_ is None:
            raise ValueError("Модель BlendingClassifier ещё не обучена.")

        meta_features = []
        for est in self.estimators:
            proba = est.predict_proba(X)
            if self.n_classes_ == 2:
                meta_features.append(proba[:, 1:2])
            else:
                k = min(9, proba.shape[1])
                meta_features.append(proba[:, :k])

        meta_X = np.hstack(meta_features)
        return meta_X

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов с использованием базовых моделей,
        передает их метамодели для получения финального предсказания.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания вероятностей классов.

        Возвращает:
        ----------
        np.array, shape (n_samples, n_classes)
            Предсказанные вероятности для каждого класса.
        """

        meta_X = self._build_meta_features(X)
        return self.final_estimator.predict_proba(meta_X)

    def predict(self, X):
        """
        Предсказывает метки классов на основе предсказаний базовых моделей
        и метамодели.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов.
        """
        meta_X = self._build_meta_features(X)
        return self.final_estimator.predict(meta_X)

class StackingClassifier:
    """
    Модель ансамбля методом стекинга (Stacking).

    Атрибуты:
    ----------
    estimators : list
        Список инициализированных базовых моделей.
    
    final_estimator : объект модели
        Метамодель, обучаемая на мета-признаках (предсказаниях базовых моделей).
    
    folds : int
        Количество фолдов для кросс-валидации при обучении базовых моделей.

    """
    def __init__(self, estimators=None, final_estimator=None, folds=...):
        if estimators is None:
            self.estimators = [
                LogisticRegression(
                    max_iter=1000,
                    multi_class="auto"
                ),
                SklearnRandomForestClassifier(
                    n_estimators=150,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=0
                ),
                KNeighborsClassifier(
                    n_neighbors=7
                ),
            ]
        else:
            self.estimators = estimators

        if final_estimator is None:
            self.final_estimator = LogisticRegression(
                max_iter=1000,
                multi_class="auto"
            )
        else:
            self.final_estimator = final_estimator

        self.folds = 5 if folds is ... else int(folds)

        self.classes_ = None
        self.n_classes_ = None
    
    def fit(self, X, y):
        """
        Обучает базовые модели на тренировочных фолдах и использует 
        их предсказания на валидационных фолдах для обучения метамодели. 
        Применяется кросс-валидация с заданным количеством фолдов.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Тренировочные данные (признаки).
        
        y : np.array, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : Stacking
            Обученная модель.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples = X.shape[0]
        n_estimators = len(self.estimators)
        if self.n_classes_ == 2:
            features_per_est = 1
        else:
            features_per_est = 9

        meta_X = np.zeros((n_samples, n_estimators * features_per_est), dtype=float)

        skf = StratifiedKFold(
            n_splits=self.folds,
            shuffle=True,
            random_state=42,
        )

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            col_start = 0
            for base_est in self.estimators:
                est = clone(base_est)
                est.fit(X_train, y_train)
                proba = est.predict_proba(X_val)

                if self.n_classes_ == 2:
                    meta_X[val_idx, col_start:col_start + 1] = proba[:, 1:2]
                    col_start += 1
                else:
                    k = min(9, proba.shape[1])
                    meta_X[val_idx, col_start:col_start + k] = proba[:, :k]
                    col_start += features_per_est

        if self.n_classes_ > 2:
            self.final_estimator = SklearnRandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
            )
        self.final_estimator.fit(meta_X, y)
        for base_est in self.estimators:
            base_est.fit(X, y)

        self.classes_ = self.final_estimator.classes_
        return self

    def _build_meta_features(self, X):
        """
        Строит мета-признаки для произвольного X с помощью
        базовых моделей, обученных на всём датасете.
        """
        if self.n_classes_ is None:
            raise ValueError("Модель StackingClassifier ещё не обучена.")

        X = np.asarray(X)
        n_samples = X.shape[0]
        n_estimators = len(self.estimators)

        if self.n_classes_ == 2:
            features_per_est = 1
        else:
            features_per_est = 9

        meta_X = np.zeros((n_samples, n_estimators * features_per_est), dtype=float)

        col_start = 0
        for est in self.estimators:
            proba = est.predict_proba(X)
            if self.n_classes_ == 2:
                meta_X[:, col_start:col_start + 1] = proba[:, 1:2]
                col_start += 1
            else:
                k = min(9, proba.shape[1])
                meta_X[:, col_start:col_start + k] = proba[:, :k]
                col_start += features_per_est  # 9

        return meta_X

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов с помощью базовых моделей, 
        передает их метамодели для получения финального предсказания.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания вероятностей классов.

        Возвращает:
        ----------
        np.array, shape (n_samples, n_classes)
            Предсказанные вероятности для каждого класса.
        """
        meta_X = self._build_meta_features(X)
        return self.final_estimator.predict_proba(meta_X)
    
    def predict(self, X):
        """
        Предсказывает метки классов на основе предсказаний базовых моделей 
        и метамодели.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов.
        """
        meta_X = self._build_meta_features(X)
        return self.final_estimator.predict(meta_X)

def softmax(x):
    """
    Вычисляет softmax функцию для входного массива x.
    
    Softmax функция преобразует входные значения в вероятности, распределяя
    их таким образом, что их сумма равна 1. Это полезно в задачах классификации,
    где требуется получить вероятности принадлежности к каждому классу.
    
    Параметры:
    ----------
    x : numpy.ndarray
        Входной массив значений размером (n_samples, n_classes), для которых необходимо вычислить softmax.
    
    Возвращает:
    ----------
    numpy.ndarray
        Массив значений softmax, где каждый элемент является вероятностью, и сумма всех элементов равна 1.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x_max = np.max(x)
        e = np.exp(x - x_max)
        return e / np.sum(e)
    else:
        x_max = np.max(x, axis=1, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=1, keepdims=True)

def one_hot_encode(y, n_classes=None):
    """
    Выполняет one-hot кодирование для заданного списка меток.

    Параметры:
    ----------
    y : numpy.ndarray или list
        Вектор или список меток классов, которые необходимо закодировать.
        Значения меток должны быть целыми числами от 0 до n_classes-1.

    n_classes : int или None, по умолчанию None
        Количество классов (размерность выходного пространства).
        Если None, то количество классов определяется автоматически как максимум значения в y плюс один.

    Возвращает:
    ----------
    numpy.ndarray
        Массив размером (n_samples, n_classes), где n_samples — количество образцов, а n_classes — количество классов.
        Каждая строка представляет собой one-hot закодированное представление соответствующей метки из y.
    """
    y = np.asarray(y, dtype=int)
    n_samples = y.shape[0]
    if n_classes is None:
        n_classes = int(np.max(y)) + 1

    one_hot = np.zeros((n_samples, n_classes), dtype=float)
    one_hot[np.arange(n_samples), y] = 1.0
    return one_hot

class BoostingClassifier:
    """
    Модель Бустинга (BoostingClassifier).

    Атрибуты:
    ----------
    n_estimators : int
        Количество деревьев в ансамбле.
    
    bootstrap : bool
        Используется ли бутстрап при построении деревьев. 
        Если False, то для каждого дерева используется весь набор данных.
    
    estimators : list
        Список деревьев с заданными параметрами (**kwargs).
    
    kwargs : dict
        Параметры для каждого дерева, такие как min_samples_split, max_depth и др.
        Передаются в DecisionTreeRegressor.
    """
    def __init__(self, n_estimators=..., bootstrap=..., lr=..., kwargs=...):
        self.n_estimators = 220 if n_estimators is ... else int(n_estimators)
        self.bootstrap = True if bootstrap is ... else bool(bootstrap)
        self.lr = 0.1 if lr is ... else float(lr)

        if kwargs is ... or kwargs is None:
            self.kwargs = {
                "max_depth": 3,
                "min_samples_leaf": 20,
                "random_state": 0,
            }
        else:
            self.kwargs = dict(kwargs)

        self.estimators = []
        self.classes_ = None
        self.n_classes_ = None
    
    def fit(self, X, y):
        """
        Обучает ансамбль на тренировочной выборке (X, y). 
        Если bootstrap=False, используется весь набор данных для каждого дерева,
        если bootstrap=True, используются бутстрап-выборки.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Тренировочные данные (признаки).
        
        y : np.array, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : BoostingClassifier
            Обученная модель.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        n_samples = X.shape[0]

        Y_one_hot = one_hot_encode(y_encoded, n_classes=self.n_classes_)
        logits = np.zeros((n_samples, self.n_classes_), dtype=float)

        self.estimators = []

        for m in range(self.n_estimators):
            probs = softmax(logits)
            neg_grad = Y_one_hot - probs

            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)

            X_boot = X[indices]
            grad_boot = neg_grad[indices]

            tree_params = dict(self.kwargs)
            if "random_state" not in self.kwargs:
                tree_params["random_state"] = m

            tree = DecisionTreeRegressor(**tree_params)
            tree.fit(X_boot, grad_boot)
            self.estimators.append(tree)

            logits += self.lr * tree.predict(X)

        return self
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        n_samples = X.shape[0]

        Y_one_hot = one_hot_encode(y_encoded, n_classes=self.n_classes_)
        logits = np.zeros((n_samples, self.n_classes_), dtype=float)
        self.estimators = []

        for m in range (self.n_estimators):
            probs = softmax(logits)
            neg_grad = Y_one_hot - probs
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            X_boot = X[indices]
        for m in range (self.n_estimators):
            probs = softmax(logits)
            neg_grad = Y_one_hot - probs
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)

            X_boot = X[indices]
            grad_boot = neg_grad[indices]

            tree_params = dict(self.kwargs)
            if "random_state" not in self.kwargs:
                tree_params["random_state"] = m

            tree = DecisionTreeRegressor(**tree_params)
            tree.fit(X_boot, grad_boot)
            self.estimators.append(tree)

            logits += self.lr * tree.predict(X)

            logits += self.lr * tree.predict(X)
        return self

    def _compute_logits(self, X):
        """
        Восстанавливает логиты для X, прогоняя все деревья.
        """
        if not self.estimators:
            raise ValueError("Модель BoostingClassifier ещё не обучена.")

        X = np.asarray(X)
        n_samples = X.shape[0]
        logits = np.zeros((n_samples, self.n_classes_), dtype=float)

        for tree in self.estimators:
            logits += self.lr * tree.predict(X)

        return logits
        

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов для каждого объекта входной выборки.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания вероятностей классов.

        Возвращает:
        ----------
        np.array, shape (n_samples, n_classes)
            Вероятности для каждого класса и каждого объекта.
        """
        logits = self._compute_logits(X)
        return softmax(logits)
        
    def predict(self, X):
        """
        Предсказывает метки классов для каждого объекта входной выборки.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов.
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

