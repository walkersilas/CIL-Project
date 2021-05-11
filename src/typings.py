"""Contains custom Python typings specific to this project."""
from abc import ABC

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin


class BaseRegressor(ABC, RegressorMixin, BaseEstimator):
    """Convenient type for a base regressor.

    This can be used as a parent class for creating custom regressors.
    """


class BaseClassifier(ABC, ClassifierMixin, BaseEstimator):
    """Convenient type for a base classifier.

    This can be used as a parent class for creating custom classifiers.
    """


class BaseTransformer(ABC, TransformerMixin, BaseEstimator):
    """Convenient type for a base data transformer.

    This can be used as a parent class for creating custom classifiers.
    """