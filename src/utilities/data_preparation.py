import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import surprise


def load_data(file_path: str,
              full_dataset: bool,
              train_val_split: bool,
              random_seed: int = 0,
              train_size: float = 0):
    data_pd = pd.read_csv(file_path)

    # Reduce Dataset for Testing
    if not full_dataset:
        data_pd = data_pd.head(10000)

    if train_val_split:
        train_pd, val_pd = train_test_split(data_pd,
                                            train_size=train_size,
                                            random_state=random_seed)
        return train_pd, val_pd
    else:
        return data_pd


def load_reinforcements(file_path: str):
    data_pd = pd.read_csv(file_path)
    return data_pd.Reinforcement.values


def __extract_users_items_predictions(data_pd: pd.DataFrame):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def __get_tensors_from_dataframe(data_pd: pd.DataFrame):
    users, movies, predictions = __extract_users_items_predictions(data_pd)
    users_torch = torch.tensor(users, dtype=torch.int64)
    movies_torch = torch.tensor(movies, dtype=torch.int64)
    predictions_torch = torch.tensor(predictions, dtype=torch.int64)

    return users_torch, movies_torch, predictions_torch


def create_dataset(data_pd: pd.DataFrame, test_dataset: bool = False):
    users_torch, movies_torch, predictions_torch = __get_tensors_from_dataframe(
        data_pd)

    if not test_dataset:
        return TensorDataset(users_torch, movies_torch, predictions_torch)
    else:
        test_ids = data_pd.Id
        return test_ids, TensorDataset(users_torch, movies_torch)


def create_dataset_with_reinforcements(data_pd: pd.DataFrame, reinforcements: np.array, test_dataset: bool = False):
    users_torch, movies_torch, ratings_torch = __get_tensors_from_dataframe(data_pd)
    reinforcements_torch = torch.tensor(reinforcements, dtype=torch.float)

    if not test_dataset:
        return TensorDataset(users_torch, movies_torch, reinforcements_torch, ratings_torch)
    else:
        test_ids = data_pd.Id
        return test_ids, TensorDataset(users_torch, movies_torch, reinforcements_torch)


def create_laplacian_matrix(data_pd: pd.DataFrame, number_of_users: int,
                            number_of_movies: int):
    users_torch, movies_torch, predictions_torch = __get_tensors_from_dataframe(
        data_pd)

    user_item_matrix = torch.sparse_coo_tensor(
        torch.vstack((users_torch, movies_torch)), predictions_torch)
    top_zero_matrix = torch.zeros(
        (user_item_matrix.shape[0], user_item_matrix.shape[0])).to_sparse()
    bottom_zero_matrix = torch.zeros(
        (user_item_matrix.shape[1], user_item_matrix.shape[1])).to_sparse()

    top_a = torch.cat((top_zero_matrix, user_item_matrix), dim=1)
    bottom_a = torch.cat(
        (torch.transpose(user_item_matrix, 0, 1), bottom_zero_matrix), dim=1)
    matrix_a = torch.vstack((top_a, bottom_a))

    degree = (matrix_a.to_dense() > 0).sum(axis=1)
    degree_matrix = torch.diag(torch.pow(degree, -0.5))

    laplacian_matrix = torch.sparse.mm(
        degree_matrix, torch.sparse.mm(matrix_a, degree_matrix)).to_sparse()
    return laplacian_matrix


def create_surprise_data(data_pd):
    users, movies, ratings = __extract_users_items_predictions(data_pd)

    df = pd.DataFrame({'users': users, 'movies': movies, 'ratings': ratings})
    reader = surprise.Reader(rating_scale=(1, 5))
    return surprise.Dataset.load_from_df(df[['users', 'movies', 'ratings']],
                                         reader=reader)
