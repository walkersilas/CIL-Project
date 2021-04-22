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

comet_api_key_path = "../comet.json"
comet_api_key = json.load(open(comet_api_key_path))

hyper_params = {
    'batch_size': 1024,
    'num_epochs': 25,
    'embedding_size': 256,
    'learning_rate': 1e-3,
    'train_size': 0.9,
    'dropout': 0.5,
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

        # Loss function for training and evaluation
        self.loss = nn.CrossEntropyLoss()

        # Layer for one-hot encoding of the users
        self.one_hot_encoding_users = nn.Embedding(number_of_users, number_of_users)
        self.one_hot_encoding_users.data = torch.eye(number_of_users)
        # Layer for one-hot encoding of the movies
        self.one_hot_encoding_movies = nn.Embedding(number_of_movies, number_of_movies)
        self.one_hot_encoding_movies.data = torch.eye(number_of_movies)

        # Dense layers for getting embedding of users and movies
        self.embedding_layer_users = nn.Linear(number_of_users, embedding_size)
        self.embedding_layer_movies = nn.Linear(number_of_movies, embedding_size)

        # Neural network used for training on concatenation of users and movies embedding
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=2 * embedding_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=5),
            nn.Dropout(p=hyper_params['dropout'])
        )

    def training_step(self, batch, batch_idx):
        users, movies, ratings = batch

        predictions_one_hot = self(users, movies)
        loss = self.loss(predictions_one_hot, ratings - 1)  # subtract 1 to receive rating in [0, 4]
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, movies, ratings = batch

        predictions_one_hot = self(users, movies)
        predictions = torch.argmax(predictions_one_hot, dim=1) + 1
        val_loss = self.loss(predictions_one_hot, ratings - 1)  # subtract 1 to receive rating in [0, 4]
        score = get_score(predictions.cpu().numpy(), ratings.cpu().numpy())
        self.log('val_loss', val_loss)
        self.log('score', score)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=hyper_params['learning_rate'])
        return optimizer

    def forward(self, users, movies):
        # Transform users and movies to one-hot encodings
        users_one_hot = self.one_hot_encoding_users(users)
        movies_one_hot = self.one_hot_encoding_movies(movies)

        # Compute embedding of users and movies
        users_embedding = self.embedding_layer_users(users_one_hot)
        movies_embedding = self.embedding_layer_movies(movies_one_hot)

        # Train rest of neural network on concatenation of user and movie embeddings
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return self.feed_forward(concat)



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

    trainer = pl.Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                         max_epochs=hyper_params['num_epochs'],
                         logger=comet_logger)

    trainer.fit(ncf, train_loader, test_loader)


if __name__ == '__main__':
    main()