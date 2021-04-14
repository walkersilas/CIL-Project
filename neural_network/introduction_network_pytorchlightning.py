import json
from comet_ml import Experiment

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning.utilities.seed
pytorch_lightning.utilities.seed.seed_everything(7)
torch.manual_seed(7)
np.random.seed(7)

comet_api_key_path = "../../../API_Keys/comet.json"
comet_api_key = json.load(open(comet_api_key_path))

hyper_params = {
    'batch_size': 1024,
    'num_epochs': 25,
    'embedding_size': 256,
    'learning_rate': 1e-3,
    'train_size': 0.9,
    'reduce_dataset': False,
    'train_data_path': '../data/data_train.csv',
    'test_data_path': '../data/data_test.csv',
    'number_of_users': 10000,
    'number_of_movies': 1000
}


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def get_score(predictions, target_values):
    return rmse(predictions, target_values)


def load_data(file_path):
    data_pd = pd.read_csv(file_path)

    if hyper_params['reduce_dataset']:
        data_pd = data_pd.head(10000)

    print(data_pd.head(5))
    print()
    print('Shape', data_pd.shape)

    train_pd, test_pd = train_test_split(data_pd, train_size=hyper_params['train_size'], random_state=42)
    return train_pd, test_pd


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def create_dataloader(users, movies, predictions):
    users_torch = torch.tensor(users, dtype=torch.int64)
    movies_torch = torch.tensor(movies, dtype=torch.int64)
    predictions_torch = torch.tensor(predictions, dtype=torch.int64)

    dataloader = DataLoader(
        TensorDataset(users_torch, movies_torch, predictions_torch),
        batch_size=hyper_params['batch_size']
    )

    return dataloader


class NCF(pl.LightningModule):
    def __init__(self, number_of_users, number_of_movies, embedding_size):
        super().__init__()
        self.embedding_layer_users = nn.Embedding(number_of_users, embedding_size)
        self.embedding_layer_movies = nn.Embedding(number_of_movies, embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=2 * embedding_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
            nn.ReLU()
        )

    def training_step(self, batch, batch_idx):
        users, movies, ratings = batch

        predictions = self(users, movies)
        loss = mse_loss(ratings, predictions)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, movies, ratings = batch

        predictions = self(users, movies)
        val_loss = mse_loss(ratings, predictions)
        rmse = math.sqrt(val_loss)
        self.log('val_loss', val_loss)
        self.log('rmse', rmse)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=hyper_params['learning_rate'])
        return optimizer

    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))


def main():
    comet_logger = CometLogger(
        api_key=comet_api_key["api_key"],
        project_name=comet_api_key["project_name"],
        workspace=comet_api_key["workspace"]
    )
    comet_logger.log_hyperparams(hyper_params)

    train_pd, test_pd = load_data(hyper_params['train_data_path'])
    train_users, train_movies, train_ratings = extract_users_items_predictions(train_pd)
    test_users, test_movies, test_ratings = extract_users_items_predictions(test_pd)

    train_loader = create_dataloader(train_users, train_movies, train_ratings)
    test_loader = create_dataloader(test_users, test_movies, test_ratings)

    ncf = NCF(hyper_params['number_of_users'],
              hyper_params['number_of_movies'],
              hyper_params['embedding_size'])

    trainer = pl.Trainer(gpus=1,
                         max_epochs=hyper_params['num_epochs'],
                         logger=comet_logger)

    trainer.fit(ncf, train_loader, test_loader)


if __name__ == '__main__':
    main()
