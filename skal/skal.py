
import numpy as np
from sklearn import clone
from sklearn.metrics import mean_absolute_error as MAE
import pdb

class ActiveLearner(object):
    """An active learner.
    """
    
    def __init__(self, estimator, extra=None):
        """init
        extra: densities, X_test, y_test, bagsize, numbags, y_query, X
        """
        self.estimator = clone(estimator)
        self.extra = extra

    def _get_oracle_mae(self, i, instance):
        estimator = clone(self.estimator)
        new_X = np.concatenate((self.extra['X'], [instance]), axis=0)
        new_y = np.concatenate((self.extra['y'], [self.extra['y_query'][i]]))
        estimator.fit(new_X, new_y)
        return MAE(estimator.predict(self.extra['X_test']), self.extra['y_test'])
        

    def _query(self, X, strategy):
        if strategy == "oracle":
            best_mae = 1000
            for i, instance in enumerate(X):
                mae = self._get_oracle_mae(i, instance)
                if mae < best_mae:
                    best_mae = mae
                    best_instance = instance
                    best_i = i
        return (best_i, best_instance)

    def query(self, X, strategy):
        """query
        """
        return self._query(X, strategy)[1]
        
    def argquery(self, X, strategy):
        """argquery
        """
        return self._query(X, strategy)[0]
