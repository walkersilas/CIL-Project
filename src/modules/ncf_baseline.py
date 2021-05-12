import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utilities.evaluation_functions import get_score


# Hyper Parameters used for the Model
hyper_parameters = {
    'batch_size': 1024,
    'num_epochs': 25,
    'number_of_users': 10000,
    'number_of_movies': 1000,
    'user_embedding_size': 10000,
    'movie_embedding_size': 1000,
    'learning_rate': 1e-3,
    'train_size': 0.9,
    'dropout': 0.5
}


class NCF(pl.LightningModule):
    def __init__(self, train_data, val_data, test_data, test_ids, args,
                 number_of_users=hyper_parameters['number_of_users'],
                 number_of_movies=hyper_parameters['number_of_movies'],
                 user_embedding_size=hyper_parameters['user_embedding_size'],
                 movie_embedding_size=hyper_parameters['movie_embedding_size'],
                 dropout=hyper_parameters['dropout']):

        super().__init__()

        self.args = args

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.test_ids = test_ids.to_numpy()

        # Loss function for training and evaluation
        self.loss = nn.MSELoss()

        # Layer for one-hot encoding of the users
        self.one_hot_encoding_users = nn.Embedding(number_of_users, number_of_users)
        self.one_hot_encoding_users.data = torch.eye(number_of_users)
        # Layer for one-hot encoding of the movies
        self.one_hot_encoding_movies = nn.Embedding(number_of_movies, number_of_movies)
        self.one_hot_encoding_movies.data = torch.eye(number_of_movies)

        # Dense layers for getting embedding of users and movies
        self.embedding_layer_users = nn.Linear(number_of_users, user_embedding_size)
        self.embedding_layer_movies = nn.Linear(number_of_movies, movie_embedding_size)

        # Neural network used for training on concatenation of users and movies embedding
        self.feed_forward = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=user_embedding_size + movie_embedding_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
            nn.ReLU()
        )

    def forward(self, users, movies):
        # Transform users and movies to one-hot encodings
        users_one_hot = self.one_hot_encoding_users(users)
        movies_one_hot = self.one_hot_encoding_movies(movies)

        # Compute embedding of users and movies
        users_embedding = self.embedding_layer_users(users_one_hot)
        movies_embedding = self.embedding_layer_movies(movies_one_hot)

        # Train rest of neural network on concatenation of user and movie embeddings
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))

    def training_step(self, batch, batch_idx):
        users, movies, ratings = batch

        predictions = self(users, movies)
        loss = self.loss(predictions, ratings.float())
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, movies, ratings = batch

        predictions = self(users, movies)
        val_loss = self.loss(predictions, ratings.float())
        score = get_score(predictions.cpu().numpy(), ratings.cpu().numpy())
        self.log('val_loss', val_loss)
        self.log('score', score)
        return val_loss

    def test_step(self, batch, batch_idx):
        users, movies = batch
        predictions = self(users, movies)
        return predictions

    def test_epoch_end(self, outputs):
        predictions = torch.cat(outputs, dim=0).cpu()
        prediction_output = np.stack((self.test_ids, predictions.numpy()), axis=1)

        self.logger.experiment.log_table(
            filename="predictions.csv",
            tabular_data=prediction_output,
            headers=["Id", "Prediction"]
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=hyper_parameters['learning_rate'])
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=hyper_parameters['batch_size'],
            shuffle=True,
            num_workers=self.args.dataloader_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=hyper_parameters['batch_size'],
            shuffle=False,
            num_workers=self.args.dataloader_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=hyper_parameters['batch_size'],
            shuffle=False,
            num_workers=self.args.dataloader_workers
        )
