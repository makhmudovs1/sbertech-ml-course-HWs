import numpy as np
from scipy import optimize


class BinaryEstimatorSVM:
    """
    Класс для построения модели бинарной классификации методом опорных 
    векторов путем решения прямой задачи оптимизации.

    Параметры:
    ----------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).
    
    Атрибуты:
    ---------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).

    fit_intercept : bool, по умолчанию True
        Включать ли свободный член (сдвиг) в модель.

    drop_last : bool, по умолчанию True
        Удалять ли последний неполный батч из обучения.

    coef_ : numpy.ndarray или None
        Коэффициенты (веса) модели размером (n_features, 1), которые обучаются на данных. Инициализируются как None до вызова метода `fit`.

    intercept_ : numpy.ndarray или None
        Свободный член (сдвиг) модели размером (1). Инициализируется как None до вызова метода `fit`.

    n_classes_ : int
        Количество классов.

    """

    def __init__(self, lr=0.01, C=1.0, n_epochs=100, batch_size=16, fit_intercept=True, drop_last=True):
        """
        Инициализация объекта класса LinearPrimalSVM с заданными гиперпараметрами.

        Параметры:
        ----------
        lr : float, default=0.01
            Скорость обучения (learning rate) для обновления коэффициентов модели.

        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

        n_epochs : int, default=100
            Количество эпох для обучения модели.

        batch_size : int, default=16
            Размер батча для mini-batch градиентного спуска (mini-batch GD).

        fit_intercept : bool, по умолчанию True
            Включать ли свободный член (сдвиг) в модель.

        drop_last : bool, по умолчанию True
            Удалять ли последний неполный батч из обучения.
        """

        self.lr = lr
        self.C = C
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.drop_last = drop_last
        self.coef_ = None
        self.intercept_ = None
        self.n_classes_ = None

    def _init_params(self, n_features):
        if self.coef_ is None:
            self.coef_ = np.zeros((n_features,), dtype=np.float64)
        if self.fit_intercept and self.intercept_ is None:
            self.intercept_ = np.array(0.0, dtype=np.float64)
        if not self.fit_intercept:
            self.intercept_ = None

    def predict(self, X):
        """
        Предсказывает расстояние до разделяющей классы гиперплоскости для входных данных на основе обученной модели.

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        numpy.ndarray
            Вектор предсказанных расстояний, ориентированных по нормали к разделяющей гиперплоскости.
        """
        X = np.asarray(X, dtype=np.float64)
        self._init_params(X.shape[1])
        w = self.coef_.reshape(-1)
        s = X @ w
        if self.fit_intercept:
            s = s + float(self.intercept_)
        return s.reshape(-1, 1)

    def loss(self, X, y_true):
        """
        Вычисляет функцию потерь для бинарной классификации на основе HingeLoss
        с учетом L2 регуляризации

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        y_true : numpy.ndarray
            Вектор истинных меток классов.

        Возвращает:
        ----------
        float
            Значение функции потерь.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)
        scores = self.predict(X).reshape(-1)
        margins = 1.0 - y * scores
        hinge = np.maximum(0.0, margins)
        w = self.coef_.reshape(-1)
        reg = 0.5 * np.dot(w, w)
        return reg + self.C * np.mean(hinge)
      
    def loss_grad(self, X, y_true):
        """
        Вычисляет градиент функции потерь по отношению к весам модели.

        В случае использования регуляризации, градиент включает соответствующие компоненты для
        штрафа за большие значения весов.

        Параметры:
        ----------

        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        y_true : numpy.ndarray
            Вектор истинных меток классов.

        Возвращает:
        ----------
        grad : numpy.ndarray
            Градиент функции потерь по отношению к весам модели.

        grad_intercept : numpy.ndarray
            Градиент функции потерь по отношению к свободному члену.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)
        scores = self.predict(X).reshape(-1)
        margins = 1.0 - y * scores
        mask = margins > 0.0
        orig_shape = self.coef_.shape
        w = self.coef_.reshape(-1)

        if np.any(mask):
            xy_sum = np.sum(y[mask, None] * X[mask], axis=0)
            grad_loss_w = -xy_sum
            grad_loss_b = -np.sum(y[mask])
        else:
            grad_loss_w = np.zeros_like(w)
            grad_loss_b = 0.0

        grad_w = w + self.C * grad_loss_w
        grad_w = grad_w.reshape(orig_shape)
        grad_b = grad_loss_b if self.fit_intercept else None
        return grad_w, grad_b

    def step(self, grad, grad_intercept):
        """
        Выполняет один шаг обновления весов модели с использованием вычисленного градиента.

        Параметры:
        ----------
        grad : numpy.ndarray
            Градиент функции потерь по отношению к весам модели (размером как coef_).
        
        grad_intercept : numpy.ndarray или None
            Градиент функции потерь по отношению к свободному члену (размером как intercept_).
            Если fit_intercept=False, этот параметр будет равен None.

        Возвращает:
        ----------
        None
        """
        self.coef_ = self.coef_ - self.lr * grad
        if self.fit_intercept:
            self.intercept_ = self.intercept_ - self.lr * grad_intercept

    def fit(self, X, y):
        """
        Обучает модель SVM с использованием mini-batch градиентного спуска (mini-batch GD).

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Тренировочные данные.

        y : numpy.ndarray, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : LinearPrimalSVM
            Обученная модель.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self._init_params(n_features)

        idx = np.arange(n_samples)
        bs = max(1, int(self.batch_size))
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n_samples, bs):
                end = start + bs
                if end > n_samples and self.drop_last:
                    continue
                Xb = X[idx[start:end]]
                yb = y[idx[start:end]]
                if Xb.size == 0:
                    continue
                gw, gb = self.loss_grad(Xb, yb)
                self.step(gw, gb)
        return self

def one_vs_rest(y, n_classes=None):
    """
    Преобразует целевые метки в матрицу, где метки целевого класса
    принимают значение 1, а остальные метки — значение -1.

    Параметры:
    ----------
    y : numpy.ndarray или list
        Вектор или список меток классов, которые необходимо закодировать.
        Значения меток должны быть целыми числами от 0 до n_classes-1.

    n_classes : int или None, по умолчанию None
        Количество классов (размерность выходного пространства).
        Если None, то количество классов определяется автоматически как максимум значения в y плюс один.

    Возвращает:
    -----------
    numpy.ndarray
        Двумерная матрица размером (n_samples, n_classes), где для каждого образца целевой
        класс представлен значением 1, а все остальные классы имеют значение -1.

    """

    y = np.asarray(y, dtype=int)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    n = y.shape[0]
    Y = -np.ones((n, n_classes), dtype=np.float64)
    rows = np.arange(n)
    Y[rows, y] = 1.0
    return Y

class LinearPrimalSVM:
    """
    Класс для построения модели многоклассовой классификации методом опорных 
    векторов путем решения прямой задачи оптимизации.

    Параметры:
    ----------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).
    
    Атрибуты:
    ---------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).

    fit_intercept : bool, по умолчанию True
        Включать ли свободный член (сдвиг) в модель.

    drop_last : bool, по умолчанию True
        Удалять ли последний неполный батч из обучения.

    self.n_classes_ : int
        Количество классов, определяемое на основе уникальных меток в обучающем наборе данных.
        Этот параметр устанавливается после вызова метода `fit` и используется для определения 
        размерности выходного пространства модели. Он равен максимальному значению метки в данных плюс один.

    list_of_models : list
        Список, содержащий бинарные модели.
    """

    def __init__(self, lr=0.01, C=1.0, n_epochs=100, batch_size=16, fit_intercept=True, drop_last=True):
        """
        Инициализация объекта класса LinearPrimalSVM с заданными гиперпараметрами.

        Параметры:
        ----------
        lr : float, default=0.01
            Скорость обучения (learning rate) для обновления коэффициентов модели.

        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

        n_epochs : int, default=100
            Количество эпох для обучения модели.

        batch_size : int, default=16
            Размер батча для mini-batch градиентного спуска (mini-batch GD).

        fit_intercept : bool, по умолчанию True
            Включать ли свободный член (сдвиг) в модель.

        drop_last : bool, по умолчанию True
            Удалять ли последний неполный батч из обучения.
        """

        self.lr = lr
        self.C = C
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.drop_last = drop_last
        self.n_classes_ = None
        self.list_of_models = []

    def predict(self, X):
        """
        Предсказывает метки классов для входных данных на основе обученной модели.

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        numpy.ndarray
            Вектор предсказанных меток классов (значения от 0 до n_classes-1).
        """
        X = np.asarray(X, dtype=np.float64)
        scores = np.hstack([m.predict(X) for m in self.list_of_models])
        return np.argmax(scores, axis=1).astype(int)


    def fit(self, X, y):
        """
        Обучает модель SVM с использованием mini-batch градиентного спуска (mini-batch GD).

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Тренировочные данные.

        y : numpy.ndarray, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : LinearPrimalSVM
            Обученная модель.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(y.max()) + 1 if self.n_classes_ is None else int(self.n_classes_)
        Y = one_vs_rest(y, n_classes=self.n_classes_)
        self.list_of_models = []
        for k in range(self.n_classes_):
            clf = BinaryEstimatorSVM(
                lr=self.lr,
                C=self.C,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                fit_intercept=self.fit_intercept,
                drop_last=self.drop_last,
            )
            clf.fit(X, Y[:, k])
            self.list_of_models.append(clf)
        return self

def kernel_linear(x1, x2):
    """
    Линейное ядро для SVM.

    Вычисляет скалярное произведение двух векторов, что соответствует линейной 
    границе разделения в пространстве признаков.

    Параметры:
    ----------
    x1 : np.array, shape (n_features,)
        Первый вектор признаков.
    
    x2 : np.array, shape (n_features,)
        Второй вектор признаков.

    Возвращает:
    ----------
    float
        Скалярное произведение векторов x1 и x2.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    return np.array(np.dot(x1, x2), dtype=np.float64)

def kernel_poly(x1, x2, d=2):
    """
    Полиномиальное ядро для SVM.

    Вычисляет полиномиальное скалярное произведение двух векторов, 
    что позволяет моделировать нелинейные границы разделения.

    Параметры:
    ----------
    x1 : np.array, shape (n_features,)
        Первый вектор признаков.
    
    x2 : np.array, shape (n_features,)
        Второй вектор признаков.
    
    d : int, default=2
        Степень полинома.

    Возвращает:
    ----------
    float
        Полиномиальное скалярное произведение векторов x1 и x2.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    return np.array((1.0 + np.dot(x1, x2)) ** int(d), dtype=np.float64)


def kernel_rbf(x1, x2, l=1.0):
    """
    Радиально-базисное (гауссовское) ядро для SVM.

    Вычисляет расстояние между двумя векторами с использованием радиально-базисной функции (RBF),
    которая позволяет моделировать сложные нелинейные зависимости.

    Параметры:
    ----------
    x1 : np.array, shape (n_features,)
        Первый вектор признаков.
    
    x2 : np.array, shape (n_features,)
        Второй вектор признаков.
    
    l : float, default=1.0
        Параметр ширины гауссовской функции (коэффициент сглаживания).

    Возвращает:
    ----------
    float
        Значение RBF-ядра между векторами x1 и x2.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    diff = x1 - x2
    val = np.exp(-np.dot(diff, diff) / (2.0 * (float(l) ** 2)))
    return np.array(val, dtype=np.float64)

def lagrange(gramm_matrix, alpha):
    """
    Двойственная функция Лагранжа для SVM.

    Вычисляет двойственную функцию для оптимизации SVM с использованием
    заранее рассчитанной матрицы Грамма.

    Параметры:
    ----------
    gramm_matrix : np.array, shape (n_samples, n_samples)
        Матрица Грамма (значения ядер между всеми парами обучающих объектов).
    
    alpha : np.array, shape (n_samples,)
        Двойственные переменные (лямбда), используемые для оптимизации.

    Возвращает:
    ----------
    float
        Значение двойственной функции Лагранжа.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    G = np.asarray(gramm_matrix, dtype=np.float64)
    return np.array(np.sum(alpha) - 0.5 * (alpha @ (G @ alpha)), dtype=np.float64)

def lagrange_derive(gramm_matrix, alpha):
    """
    Производная двойственной функции Лагранжа по alpha.

    Вычисляет градиент (производную) двойственной функции Лагранжа,
    что необходимо для решения задачи оптимизации.

    Параметры:
    ----------
    gramm_matrix : np.array, shape (n_samples, n_samples)
        Матрица Грама (значения ядер между всеми парами обучающих объектов).
    
    alpha : np.array, shape (n_samples,)
        Двойственные переменные (лямбда), используемые для оптимизации.

    Возвращает:
    ----------
    np.array, shape (n_samples,)
        Градиент двойственной функции по alpha.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    G = np.asarray(gramm_matrix, dtype=np.float64)
    return np.asarray(np.ones_like(alpha) - G @ alpha, dtype=np.float64)

def one_vs_one(X, y, n_classes=None):
    """
    Преобразует целевые метки в матрицу, где метки первого класса
    принимают значение 1, а метки второго — значение -1.

    Параметры:
    ----------
    y : numpy.ndarray
        Вектор или список меток классов, которые необходимо закодировать.
        Значения меток должны быть целыми числами от 0 до n_classes-1.

    n_classes : int или None, по умолчанию None
        Количество классов (размерность выходного пространства).
        Если None, то количество классов определяется автоматически как максимум значения в y плюс один.

    Возвращает:
    -----------
    list of tuples
        (X_cut, y_cut (Бинарный таргет 1 или -1), соответствующий '1' класс, соответствующий '-1' класс)
        
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    res = []
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            mask  = (y == a) | (y == b)
            y_cut = (y[mask] == a).astype(np.int64) * 2 - 1
            res.append((X[mask], y_cut, a, b))
    return res


class SoftMarginSVM:
    """
    Реализация SVM с мягким зазором (Soft Margin SVM) с возможностью использовать произвольные ядра.
    
    Атрибуты:
    ----------
    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.
    
    alpha : np.array, shape (n_samples,)
        Двойственные переменные (лямбда) для решения задачи оптимизации.

    supportVectors : np.array, shape (n_support_vectors, n_features)
        Опорные вектора — обучающие объекты, которые оказывают влияние на разделяющую гиперплоскость.

    supportLabels : np.array, shape (n_support_vectors,)
        Метки классов для опорных векторов.

    supportalpha : np.array, shape (n_support_vectors,)
        Значения альфа (лямбда) для опорных векторов.

    kernel : function
        Ядро для вычисления скалярных произведений в пространстве признаков.

    classes_names : list or array-like, shape (2,)
        Имена классов. Используются для преобразования предсказанных значений {-1, 1} в имена классов.
    
    b: float 
        Смещение.
        
    """
    
    def __init__(self, C, kernel_func, classes_names=None):
        """
        Инициализирует модель Soft Margin SVM.
        
        Параметры:
        ----------
        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.
        
        kernel_func : function
            Функция ядра, определяющая метод вычисления скалярных произведений в новом пространстве признаков.
        
        classes_names : list
            Список имен классов. Ожидается, что в обучающих данных метки классов {-1, 1}.
        """
        self.C = C                                 
        self.alpha = None
        self.supportVectors = None
        self.supportLabels = None
        self.supportalpha = None
        self.kernel = kernel_func
        self.classes_names = classes_names
        self.b = None

    def _gram_y(self, X, y):
        n = X.shape[0]
        K = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])
        yy = np.outer(y, y)
        return K * yy

    def predict(self, X):
        """
        Предсказывает метки классов для входных данных.
        
        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Массив объектов для предсказания.
        
        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов, где метки соответствуют значениям из `classes_names`.
        """

        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.empty((n, 1), dtype=np.float64)
        for i in range(n):
            s = 0.0
            for a, yj, xj in zip(self.supportalpha, self.supportLabels, self.supportVectors):
                s += a * yj * self.kernel(xj, X[i])
            s += self.b
            if self.classes_names is None:
                out[i, 0] = 1.0 if s > 0 else -1.0
            else:
                out[i, 0] = float(self.classes_names[0]) if s > 0 else float(self.classes_names[1])
        return out

    def fit(self, X, y):
        """
        Обучает модель с использованием оптимизации двойственной задачи для SVM.
        
        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Обучающие данные (матрица признаков).
        
        y : np.array, shape (n_samples,)
            Вектор меток классов, должен содержать значения {-1, 1}.
        
        Возвращает:
        ----------
        self : SoftMarginSVM
            Обученная модель.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = X.shape[0]
        G = self._gram_y(X, y)
        def obj(a):
            return -lagrange(G, a)
        def grad(a):
            return -lagrange_derive(G, a)

        cons = ({'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y})
        bounds = [(0.0, self.C)] * n
        x0 = np.zeros(n, dtype=np.float64)
        res = optimize.minimize(obj, x0, jac=grad, bounds=bounds, constraints=cons,
                                method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-9})
        alpha = res.x
        eps = 1e-6
        sv_mask = alpha > eps
        self.alpha = alpha[sv_mask]
        self.supportVectors = X[sv_mask]
        self.supportLabels = y[sv_mask]
        self.supportalpha = self.alpha

        mid_mask = (alpha > eps) & (alpha < self.C - eps)
        idx = np.where(mid_mask)[0] if np.any(mid_mask) else np.where(sv_mask)[0]
        bs = []
        for k in idx:
            s = 0.0
            for aj, yj, xj in zip(self.alpha, self.supportLabels, self.supportVectors):
                s += aj * yj * self.kernel(xj, X[k])
            bs.append(y[k] - s)
        self.b = float(np.mean(bs)) if len(bs) else 0.0
        return self

class NonLinearDualSVM:
    """
    NonLinearDualSVM реализует SVM one-vs-one с использованием двойственной задачи. 
    Поддерживает использование различных ядерных функций для задач классификации.

    Атрибуты:
    ---------
    estimators : list или None
        Список бинарный SVM моделе one-vs-one

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    kernel : str, default='rbf'
        Ядерная функция, используемая в модели (возможные значения: 'poly', 'rbf', 'linear').

    """

    def __init__(self, C=1.0, kernel='rbf', kernel_parameter=1.0):
        """
        Инициализирует модель NonLinearDualSVM с указанной ядерной функцией.
        
        Параметры:
        ----------
        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

        kernel : str, default='rbf'
            Ядерная функция, используемая в модели (возможные значения: 'poly', 'rbf', 'linear').

        kernel_parameter : float, default=1.0
            Гиперпарметр ядра
      
        """
        self.C = C
        if kernel == 'poly':
          self.kernel = lambda x, y: kernel_poly(x, y, d=kernel_parameter)
        elif kernel == 'rbf':
          self.kernel = lambda x, y: kernel_rbf(x, y, l=kernel_parameter)
        else:
          self.kernel = kernel_linear
        self.kernel.__name__='kernel'
        self.estimators = []

    def fit(self, X, y):
        """
        Обучает модель SVM на тренировочных данных (X, y) с использованием двойственной задачи.
        
        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Тренировочные данные.

        y : numpy.ndarray, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        -------
        self : NonLinearDualSVM
            Обученная модель.
        """
        
        #CODE HERE
        return
    

    def predict(self, X):
        """
        Предсказывает метки классов для входных данных X.
        
        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        -------
        numpy.ndarray, shape (n_samples,)
            Предсказанные метки классов для каждого образца.
        """

        #CODE HERE
        return
