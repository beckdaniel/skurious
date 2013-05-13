
import numpy as np
from sklearn import clone

class Bagging():
    '''Wrapper for bagging'''
    def __init__(self, estimator, numbags=20, bagsize=None):
        self.estimator = estimator
        self.numbags = numbags
        self.bagsize = bagsize

    def _gen_bags(self):
        self.bags = []
        for i in range(self.numbags):
            samples = np.random.randint(self.data.shape[0],
                                        size=(self.bagsize,))
            self.bags.append(self.data[samples])
        self.bags = np.array(self.bags)

    def fit(self, X, y):
        self.data = np.concatenate((X, np.array([y]).T), axis=1)
        if self.bagsize == None:
            # default bagsize is same amount of data
            self.bagsize = self.data.shape[0]
        self._gen_bags()
        self.estimators = []
        for bag in self.bags:
            l = clone(self.estimator)
            try:
                l.fit(bag[:,:-1], bag[:,-1])
                self.estimators.append(l)
            except ValueError:
                #sometimes it fails because the bag only contains one class
                #for now we just drop the bag but need to find a better solution
                continue 

    def _predict_per_estimator(self, X):
        preds = []
        for l in self.estimators:
            preds.append(l.predict(X))
        return np.array(preds)

    def predict(self, X):
        return np.mean(self._predict_per_estimator(X), axis=0)

    def predict_vars(self, X):
        return np.std(self._predict_per_estimator(X), axis=0) ** 2

