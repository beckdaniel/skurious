
import numpy as np
from sklearn import clone
from sklearn.metrics import mean_absolute_error as MAE
import pdb
import cProfile

class ActiveLearner(object):
    """An active learner.
    """
    
    def __init__(self, estimator):
        """init
        extra: densities, X_test, y_test, y_query, X_train, y_train
        """
        self.estimator = estimator

    def __getattr__(self, attribute):
        """This is used to keep the API for the base estimator.
        """
        return getattr(self.estimator, attribute)

    def _get_oracle_mae(self, i, instance):
        estimator = clone(self.estimator)
        new_X = np.concatenate((self.extra['X_train'], [instance]), axis=0)
        new_y = np.concatenate((self.extra['y_train'], [self.extra['y_query'][i]]))
        estimator.fit(new_X, new_y)
        return MAE(estimator.predict(self.extra['X_test']), self.extra['y_test'])

    def _query_oracle(self, X_query):
        best_mae = 1000
        for i, instance in enumerate(X_query):
            mae = self._get_oracle_mae(i, instance)
            if mae < best_mae:
                best_mae = mae
                best_instance = instance
                best_i = i
        return (best_i, best_instance)
        
    def _query_us(self, X_query):
        vars = self.estimator.predict_vars(X_query)
        best_i = vars.argmax()
        best_instance = X_query[best_i]
        return (best_i, best_instance)

    def _query_id(self, X_query):
        vars = self.estimator.predict_vars(X_query)
        values = vars * np.mean(self.extra["densities"], axis=0)
        best_i = values.argmax()
        best_instance = X_query[best_i]
        return (best_i, best_instance)

    def _query_random(self, X_query):
        best_i = np.random.randint(0, high=X_query.shape[0])
        best_instance = X_query[best_i]
        return (best_i, best_instance)

    def _query(self, X_query, strategy):
        if strategy == "oracle":
            return self._query_oracle(X_query)
        elif strategy == "us":
            return self._query_us(X_query)
        elif strategy == "id":
            return self._query_id(X_query)
        elif strategy == "random":
            return self._query_random(X_query)

    def query(self, X_query, strategy, extra=None):
        self.extra = extra
        return self._query(X_query, strategy)[1]

    def argquery(self, X_query, strategy, extra=None):
        """argquery: we filter the last column
        """
        self.extra = extra
        return self._query(X_query, strategy)[0]


class GPActiveLearner(ActiveLearner):
    """A GP ActiveLearner
    """
    
    def _query_us(self, X_query):
        vars = self.estimator.predict(X_query)[1]
        best_i = vars.argmax()
        best_instance = X_query[best_i]
        return (best_i, best_instance)
        
    def _query_id(self, X_query):
        vars = self.estimator.predict(X_query)[1]
        values = np.ndarray.flatten(vars) * np.mean(self.extra["densities"], axis=0)
        best_i = values.argmax()
        best_instance = X_query[best_i]
        return (best_i, best_instance)
