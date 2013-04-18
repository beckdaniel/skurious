
import numpy as np
from ..skurious.skurious import ActiveLearner

class MockMeanEstimator(object):

    def __init__(self):
        self.params = {}

    def fit(self, X, y):
        self.mean = np.mean(y)

    def predict(self, X_test):
        return np.array([self.mean] * X_test.shape[0])

    def get_params(self, deep=False):
        return self.params


def test_oracle_strategy_1():
    X = np.random.randint(100, size=(10,5))
    y = np.array([7] * 10)
    X_query = np.random.randint(100, size=(10,5))
    est = MockMeanEstimator()
    extra = {}
    extra['y_query'] = np.array(range(10))
    extra['X_test'] = np.random.randint(100, size=(5,5))
    extra['y_test'] = np.array([7] * 5)
    extra['X_train'] = X
    extra['y_train'] = y
    al = ActiveLearner(est, extra)
    al.estimator.fit(X, y)
    q = al.query(X_query, "oracle")
    assert (q == X_query[7]).all()


def test_oracle_strategy_2():
    X = np.random.randint(100, size=(10,5))
    y = np.array([7] * 10)
    X_query = np.random.randint(100, size=(10,5))
    est = MockMeanEstimator()
    extra = {}
    extra['y_query'] = np.array(range(10))
    extra['X_test'] = np.random.randint(100, size=(5,5))
    extra['y_test'] = np.array([6] * 5)
    extra['X_train'] = X
    extra['y_train'] = y
    al = ActiveLearner(est, extra)
    al.estimator.fit(X, y)
    q = al.query(X_query, "oracle")
    assert (q == X_query[0]).all()


def test_oracle_strategy_3():
    X = np.random.randint(100, size=(10,5))
    y = np.array([7] * 10)
    X_query = np.random.randint(100, size=(10,5))
    est = MockMeanEstimator()
    extra = {}
    extra['y_query'] = np.array(range(10))
    extra['X_test'] = np.random.randint(100, size=(5,5))
    extra['y_test'] = np.array([8] * 5)
    extra['X_train'] = X
    extra['y_train'] = y
    al = ActiveLearner(est, extra)
    al.estimator.fit(X, y)
    q = al.query(X_query, "oracle")
    assert (q == X_query[9]).all()


def test_oracle_strategy_4():
    X = np.random.randint(100, size=(10,5))
    y = np.array([7] * 10)
    X_query = np.random.randint(100, size=(10,5))
    print X_query.shape
    print np.array(range(10)).shape
    X_query = np.concatenate((X_query, np.array([range(10)]).T), axis=1)
    est = MockMeanEstimator()
    extra = {}
    extra['y_query'] = np.array(range(10))
    extra['X_test'] = np.random.randint(100, size=(5,5))
    extra['y_test'] = np.array([7] * 5)
    extra['X_train'] = X
    extra['y_train'] = y
    al = ActiveLearner(est, extra)
    al.estimator.fit(X, y)
    q = al.argquery(X_query, "oracle")
    assert q == 7


def test_oracle_strategy_5():
    X = np.random.randint(100, size=(10,5))
    y = np.array([7] * 10)
    X_query = np.random.randint(100, size=(10,5))
    X_query = np.concatenate((X_query, np.array([range(10)]).T), axis=1)
    est = MockMeanEstimator()
    extra = {}
    extra['y_query'] = np.array(range(10))
    extra['X_test'] = np.random.randint(100, size=(5,5))
    extra['y_test'] = np.array([4] * 5)
    extra['X_train'] = X
    extra['y_train'] = y
    al = ActiveLearner(est, extra)
    al.estimator.fit(X, y)
    q = al.argquery(X_query, "oracle")
    assert q == 0
