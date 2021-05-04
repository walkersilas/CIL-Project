import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def load_data(file_path: str, full_dataset: bool, train_val_split: bool, random_seed: int = 0, train_size: float = 0):
    data_pd = pd.read_csv(file_path)

    # Reduce Dataset for Testing
    if not full_dataset:
        data_pd = data_pd.head(10000)

    if train_val_split:
        train_pd, val_pd = train_test_split(data_pd, train_size=train_size, random_state=random_seed)
        return train_pd, val_pd
    else:
        return data_pd


def __extract_users_items_predictions(data_pd: pd.DataFrame):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def create_dataset(data_pd: pd.DataFrame, test_dataset: bool = False):
    users, movies, predictions = __extract_users_items_predictions(data_pd)
    users_torch = torch.tensor(users, dtype=torch.int64)
    movies_torch = torch.tensor(movies, dtype=torch.int64)
    predictions_torch = torch.tensor(predictions, dtype=torch.int64)

    if not test_dataset:
        return TensorDataset(users_torch, movies_torch, predictions_torch)
    else:
        test_ids = data_pd.Id
        return test_ids, TensorDataset(users_torch, movies_torch)
