import math
from sklearn.metrics import mean_squared_error
import torch


def get_score(predictions, target_values):
    mse = mean_squared_error(predictions, target_values)
    return math.sqrt(mse)


def weighted_mse_loss(predictions, target_values, weights):
    return torch.sum(weights * (predictions - target_values) ** 2)
