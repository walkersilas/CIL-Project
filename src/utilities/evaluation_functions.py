import math
from sklearn.metrics import mean_squared_error


def get_score(predictions, target_values):
    mse = mean_squared_error(predictions, target_values)
    return math.sqrt(mse)
