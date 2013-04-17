

class ActiveLearner(object):
    """An active learner.
    """
    
    def __init__(self, estimator, extra=None):
        """init
        extra: densities, X_test, y_test
        """
        self.estimator = estimator
        self.extra = extra

    def _query(self, X, strategy):
        if strategy = "random":
            pass

    def query(self, X, strategy):
        """query
        """
        return self._query(X, strategy)[1]
        
    def argquery(self, X, strategy):
        """argquery
        """
        return self._query(X, strategy)[0]
