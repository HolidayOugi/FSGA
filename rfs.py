import random
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class RandomFeatureSelection(TransformerMixin, BaseEstimator):
    def __init__(self, fc=10):
        self.random_features = []
        self.features_count = fc

    def fit(self, X):
        feature_list = list(X.columns.values)
        self.random_features = random.sample(feature_list, self.features_count)  # random.choice returns duplicates

    def transform(self, X):
        X = X.filter(self.random_features)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        X = self.transform(X)
        return X
