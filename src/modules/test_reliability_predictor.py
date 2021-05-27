import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utilities.evaluation_functions import get_score


class RELIABILITY_PREDICTOR(pl.LightningModule):
    def __init__(self, train_data, val_data, test_data, args, config):

        super().__init__()

        self.args = args

        # Configuration used for execution
        self.config = config

        # Parameters of the network
        self.number_of_users = config['number_of_users']
        self.number_of_movies = config['number_of_movies']
        self.user_embedding_size = config['user_embedding_size']
        self.movie_embedding_size = config['movie_embedding_size']
        self.dropout = config['dropout']

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # Loss function for training and evaluation
        self.loss = nn.MSELoss()

        # Layer for one-hot encoding of the users
        self.one_hot_encoding_users = nn.Embedding(self.number_of_users, self.number_of_users)
        self.one_hot_encoding_users.data = torch.eye(self.number_of_users)
        # Layer for one-hot encoding of the movies
        self.one_hot_encoding_movies = nn.Embedding(self.number_of_movies, self.number_of_movies)
        self.one_hot_encoding_movies.data = torch.eye(self.number_of_movies)

        # Dense layers for getting embedding of users and movies
        self.embedding_layer_users = nn.Linear(self.number_of_users, self.user_embedding_size)
        self.embedding_layer_movies = nn.Linear(self.number_of_movies, self.movie_embedding_size)

        # Neural network used for training on concatenation of users and movies embedding
        self.feed_forward = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.user_embedding_size + self.movie_embedding_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
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
        users, movies, reliabilities, _ = batch

        predictions = self(users, movies)
        loss = self.loss(reliabilities, predictions.float())
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, movies, reliabilities, ratings = batch

        predictions = self(users, movies)
        val_loss = self.loss(predictions, reliabilities.float())
        score = get_score(predictions.cpu().numpy(), reliabilities.cpu().numpy())
        self.log('val_loss', val_loss)
        self.log('score', score)
        return val_loss

    def test_step(self, batch, batch_idx):
        users, movies = batch
        predictions = self(users, movies)
        return predictions

    def test_epoch_end(self, outputs):
        predictions = torch.cat(outputs, dim=0).cpu()

        return {'predictions': predictions.numpy()}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.args.dataloader_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.args.dataloader_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.args.dataloader_workers
        )
