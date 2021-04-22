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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

comet_api_key_path = "../comet.json"
comet_api_key = json.load(open(comet_api_key_path))

hyper_params = {
    'batch_size': 1024,
    'num_epochs': 25,
    'encoding_size': 256,
    'learning_rate': 1e-3,
    'train_size': 0.9,
    'dropout': 0.5,
    'reduce_dataset': False,
    'train_data_path': '../data/data_train.csv',
    'sample_submission_path': '../data/sampleSubmission.csv',
    'number_of_users': 10000,
    'number_of_movies': 1000
}


def masked_mse_loss(predictions, target, mask):
    return torch.mean(mask * (predictions - target) ** 2)


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def get_score(predictions, target_values):
    return rmse(predictions, target_values)


def load_data(file_path, submission=False):
    data_pd = pd.read_csv(file_path)

    if hyper_params['reduce_dataset']:
        data_pd = data_pd.head(10000)

    print(data_pd.head(5))
    print()
    print('Shape', data_pd.shape)

    if submission:
        return data_pd, None
    else:
        train_pd, test_pd = train_test_split(data_pd, train_size=hyper_params['train_size'], random_state=42)
        return train_pd, test_pd


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def get_full_matrices(users, movies, ratings):
    shape = (hyper_params['number_of_users'], hyper_params['number_of_movies'])
    data, mask = np.zeros(shape), np.zeros(shape)

    for user, movie, rating in zip(users, movies, ratings):
        data[user - 1][movie - 1] = rating
        mask[user - 1][movie - 1] = 1

    return data, mask


def create_train_dataloader(data, mask):
    data_torch = torch.tensor(data, dtype=torch.int64)
    mask_torch = torch.tensor(mask, dtype=torch.int64)

    dataloader = DataLoader(
        TensorDataset(data_torch, mask_torch),
        batch_size=hyper_params['batch_size']
    )
    return dataloader


def create_test_dataloader(users, movies, ratings):
    users_torch = torch.tensor(users, dtype=torch.int64)
    movies_torch = torch.tensor(movies, dtype=torch.int64)
    ratings_torch = torch.tensor(ratings, dtype=torch.int64)

    dataloader = DataLoader(
        TensorDataset(users_torch, movies_torch, ratings_torch),
        batch_size=hyper_params['batch_size']
    )
    return dataloader


def create_evaluation_dataloader(users, movies):
    users_torch = torch.tensor(users, dtype=torch.int64)
    movies_torch = torch.tensor(movies, dtype=torch.int64)

    dataloader = DataLoader(
        TensorDataset(users_torch, movies_torch),
        batch_size=hyper_params['batch_size']
    )
    return dataloader


class AutoEncoder(pl.LightningModule):
    def __init__(self, number_of_users, number_of_movies, encoding_size, all_data):
        super().__init__()

        # Tensor containing all the users of the data
        self.all_data = all_data

        # Loss function for training
        self.loss = masked_mse_loss

        # Encoder part of the Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=number_of_users, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=encoding_size),
            nn.ReLU()
        )

        # Decoder part of the Autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=encoding_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=number_of_users),
            nn.ReLU()
        )

    def training_step(self, batch, batch_idx):
        data, mask = batch

        reconstructions = self(data.float())
        loss = self.loss(reconstructions, data.float(), mask)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, movies, ratings = batch

        reconstructed_matrix = self.reconstruct_full_matrix()
        predictions = self.extract_prediction_from_full_matrix(reconstructed_matrix, users, movies)
        score = get_score(predictions.cpu().numpy(), ratings.cpu().numpy())
        self.log('score', score)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=hyper_params['learning_rate'])
        return optimizer

    def forward(self, data):
        return self.decoder(self.encoder(data))

    def reconstruct_full_matrix(self):
        reconstructed_matrix = torch.zeros((hyper_params['number_of_movies'], hyper_params['number_of_users']))

        for i in range(0, hyper_params['number_of_movies'], hyper_params['batch_size']):
            upper_bound = min(i + hyper_params['batch_size'], hyper_params['number_of_movies'])
            reconstruction = self(self.all_data[i:upper_bound].float().to(device))
            reconstructed_matrix[i:upper_bound] = reconstruction
        return reconstructed_matrix.T

    def extract_prediction_from_full_matrix(self, reconstructed_matrix, users, movies):
        predictions = torch.zeros(len(users))

        for i, (user, movie) in enumerate(zip(users, movies)):
            predictions[i] = reconstructed_matrix[user][movie]
        return predictions


def main():
    comet_logger = CometLogger(
        api_key=comet_api_key["api_key"],
        project_name=comet_api_key["project_name"],
        workspace=comet_api_key["workspace"],
        disabled=hyper_params['reduce_dataset']
    )

    comet_logger.log_hyperparams(hyper_params)

    train_pd, test_pd = load_data(hyper_params['train_data_path'])
    train_users, train_movies, train_ratings = extract_users_items_predictions(train_pd)
    test_users, test_movies, test_ratings = extract_users_items_predictions(test_pd)
    train_data, train_mask = get_full_matrices(train_users, train_movies, train_ratings)
    train_data, train_mask = train_data.T, train_mask.T

    all_data_pd, _ = load_data(hyper_params['train_data_path'])
    all_users, all_movies, all_ratings = extract_users_items_predictions(all_data_pd)
    all_data, _ = get_full_matrices(all_users, all_movies, all_ratings)
    all_data = all_data.T

    train_loader = create_train_dataloader(train_data, train_mask)
    test_loader = create_test_dataloader(test_users, test_movies, test_ratings)

    autoencoder = AutoEncoder(hyper_params['number_of_users'],
                              hyper_params['number_of_movies'],
                              hyper_params['encoding_size'],
                              torch.tensor(all_data, dtype=torch.int64))

    trainer = pl.Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                         max_epochs=(hyper_params['num_epochs'] if not hyper_params['reduce_dataset'] else 1),
                         logger=comet_logger)

    trainer.fit(autoencoder, train_loader, test_loader)


if __name__ == '__main__':
    main()
