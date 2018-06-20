from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = Pipeline([
            ('imputer', Imputer(strategy='median')),
            ('regressor', linear_model.LinearRegression())
        ])

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
