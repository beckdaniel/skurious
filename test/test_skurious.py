
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


def preprocess_oracle(y_test):
    X = np.random.randint(100, size=(10,5))
    y = np.array([7] * 10)
    X_query = np.random.randint(100, size=(10,5))
    est = MockMeanEstimator()
    extra = {}
    extra['y_query'] = np.array(range(10))
    extra['X_test'] = np.random.randint(100, size=(5,5))
    extra['y_test'] = np.array([y_test] * 5)
    extra['X_train'] = X
    extra['y_train'] = y
    al = ActiveLearner(est, extra)
    al.fit(X, y)
    return (al, X_query)


def test_oracle_strategy_1():
    al, X_query = preprocess_oracle(7)
    q = al.query(X_query, "oracle")
    assert (q == X_query[7]).all()


def test_oracle_strategy_2():
    al, X_query = preprocess_oracle(6)
    q = al.query(X_query, "oracle")
    assert (q == X_query[0]).all()


def test_oracle_strategy_3():
    al, X_query = preprocess_oracle(8)
    q = al.query(X_query, "oracle")
    assert (q == X_query[9]).all()


def test_oracle_strategy_4():
    al, X_query = preprocess_oracle(7)#, arg=True)
    q = al.argquery(X_query, "oracle")
    assert q == 7


def test_oracle_strategy_5():
    al, X_query = preprocess_oracle(4)#, arg=True)
    q = al.argquery(X_query, "oracle")
    assert q == 0
