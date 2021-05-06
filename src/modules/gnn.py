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
    'num_epochs': 60,
    'number_of_users': 10000,
    'number_of_movies': 1000,
    'embedding_size': 64,
    'learning_rate': 1e-3,
    'train_size': 0.9,
    'patience': 3
}


class GNN(pl.LightningModule):
    def __init__(self, train_data, val_data, test_data, test_ids, args, laplacian_matrix,
                 number_of_users=hyper_parameters['number_of_users'],
                 number_of_movies=hyper_parameters['number_of_movies'],
                 embedding_size=hyper_parameters['embedding_size']):

        super().__init__()

        self.args = args

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.test_ids = test_ids.to_numpy()

        # Loss function for training and evaluation
        self.loss = nn.MSELoss()

        # Number of users and movies
        self.number_of_users = number_of_users
        self.number_of_movies = number_of_movies

        # Layers for embedding users and movies
        self.embedding_users = nn.Embedding(number_of_users, embedding_size)
        self.embedding_movies = nn.Embedding(number_of_movies, embedding_size)

        # Laplacian and Identity Matrices for the Embedding Propagation Layers
        self.laplacian_matrix = laplacian_matrix.to(self.device)
        self.identity = torch.eye(number_of_users + number_of_movies).to_sparse()

        # List of Embedding Propagation Layers
        self.embedding_propagation_layers = torch.nn.ModuleList([
            self.EmbeddingPropagationLayers(self.laplacian_matrix, self.identity, in_features=64, out_features=64),
            self.EmbeddingPropagationLayers(self.laplacian_matrix, self.identity, in_features=64, out_features=64)
        ])
        num_embedding_propagation_layers = len(self.embedding_propagation_layers) + 1

        # Feedforward network used to make predictions from the embedding propaagtiona layers
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=64 * num_embedding_propagation_layers * 2, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=1)
        )

    def get_initial_embeddings(self):
        users = torch.LongTensor([i for i in range(self.number_of_users)]).to(self.device)
        movies = torch.LongTensor([i for i in range(self.number_of_movies)]).to(self.device)

        users_embedding = self.embedding_users(users)
        movies_embedding = self.embedding_movies(movies)
        return torch.cat((users_embedding, movies_embedding), dim=0)

    def forward(self, users, movies):
        current_embedding = self.get_initial_embeddings()
        final_embedding = current_embedding.clone()
        for layer in self.embedding_propagation_layers:
            current_embedding = layer(current_embedding, self.device)
            # TODO: Do we really need to include the first (current) embedding in the final embedding?
            final_embedding = torch.cat((final_embedding, current_embedding), dim=1)

        users_embedding = final_embedding[users]
        movies_embedding = final_embedding[movies + self.number_of_users]
        concat = torch.cat((users_embedding, movies_embedding), dim=1)

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

    # Internal Embedding Propagation Layers for the GNN
    class EmbeddingPropagationLayers(nn.Module):
        def __init__(self, laplacian_matrix, identity, in_features, out_features):
            super().__init__()

            # Laplacian Matrix used in the Embedding Layer
            self.laplacian_matrix = laplacian_matrix

            # Identity Matrix used in the Embedding Layer
            self.identity = identity

            # Linear transformation Layers used internally
            self.transformation_layer_1 = nn.Linear(in_features, out_features)
            self.transformation_layer_2 = nn.Linear(in_features, out_features)

        def forward(self, previous_embedding, device):
            self.laplacian_matrix = self.laplacian_matrix.to(device)
            self.identity = self.identity.to(device)

            embedding_1 = torch.sparse.mm((self.laplacian_matrix + self.identity), previous_embedding)
            embedding_2 = torch.mul(torch.sparse.mm(self.laplacian_matrix, previous_embedding), previous_embedding)

            transformed_embedding_1 = self.transformation_layer_1(embedding_1)
            transformed_embedding_2 = self.transformation_layer_2(embedding_2)

            return nn.ReLU()(transformed_embedding_1 + transformed_embedding_2)
