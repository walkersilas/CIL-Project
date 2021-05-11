from sklearn.ensemble import BaggingRegressor
from sklearn.base import BaseEstimator
import numpy

def predict_with_bagging(model: BaseEstimator, train_data: numpy.ndarray, test_data: numpy.ndarray):
    regressor = (BaggingRegressor(base_estimator=model, n_estimators=3, random_state=0)
        .fit(train_data))
    regressor.predict(test_data)

    return regressor