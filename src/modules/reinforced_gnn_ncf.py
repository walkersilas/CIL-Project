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
    'num_embedding_propagation_layers': 2,
    'learning_rate': 5e-4,
    'train_size': 0.9,
    'patience': 3,
    'dropout': 0,
    'reinforcement_type': ["svd", "nmf", "slopeone"]
}


class GNN(pl.LightningModule):
    def __init__(self, train_data, val_data, test_data, test_ids, args, laplacian_matrix, config):

        super().__init__()

        self.args = args

        # Configuration used for execution
        self.config = config

        # Parameters of the network
        self.number_of_users = config['number_of_users']
        self.number_of_movies = config['number_of_movies']
        self.embedding_size = config['embedding_size']
        self.num_embedding_propagation_layers = config['num_embedding_propagation_layers']
        self.dropout = config['dropout']
        self.num_reinforcements = len(config["reinforcement_type"])

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.test_ids = test_ids.to_numpy()

        # Loss function for training and evaluation
        self.loss = nn.MSELoss()

        # Layers for embedding users and movies
        self.embedding_users = nn.Embedding(self.number_of_users, self.embedding_size)
        self.embedding_movies = nn.Embedding(self.number_of_movies, self.embedding_size)

        # Laplacian and Identity Matrices for the Embedding Propagation Layers
        self.laplacian_matrix = laplacian_matrix.to(self.device)
        self.identity = torch.eye(self.number_of_users + self.number_of_movies).to_sparse()

        # List of Embedding Propagation Layers
        self.embedding_propagation_layers = torch.nn.ModuleList([
            self.EmbeddingPropagationLayers(self.laplacian_matrix, self.identity,
                                            in_features=self.embedding_size, out_features=self.embedding_size)
            for i in range(self.num_embedding_propagation_layers)
        ])

        # Feedforward network used to make predictions from the embedding propaagtiona layers
        input_size = 2 *  self.num_embedding_propagation_layers * self.embedding_size
        self.feed_forward = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=input_size, out_features=64),
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

        # Layer combining the reinforcements with the output of the neural network
        self.combination_layer = nn.Linear(in_features=1 + self.num_reinforcements, out_features=1)

        # Initialize the weights of the linear layers
        self.feed_forward.apply(self.init_weights)
        self.init_weights(self.combination_layer)

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def get_initial_embeddings(self):
        users = torch.LongTensor([i for i in range(self.number_of_users)]).to(self.device)
        movies = torch.LongTensor([i for i in range(self.number_of_movies)]).to(self.device)

        users_embedding = self.embedding_users(users)
        movies_embedding = self.embedding_movies(movies)
        return torch.cat((users_embedding, movies_embedding), dim=0)

    def forward(self, users, movies, reinforcements):
        current_embedding = self.get_initial_embeddings()
        final_embedding = None
        for layer in self.embedding_propagation_layers:
            current_embedding = layer(current_embedding, self.device)
            if final_embedding is None:
                final_embedding = current_embedding
            else:
                final_embedding = torch.cat((final_embedding, current_embedding), dim=1)

        users_embedding = final_embedding[users]
        movies_embedding = final_embedding[movies + self.number_of_users]
        concat = torch.cat((users_embedding, movies_embedding), dim=1)
        feed_forward_output = self.feed_forward(concat)

        concat = torch.cat((feed_forward_output, reinforcements), dim=1)

        return torch.squeeze(self.combination_layer(concat))

    def training_step(self, batch, batch_idx):
        users, movies, reinforcements, ratings = batch

        predictions = self(users, movies, reinforcements)
        loss = self.loss(predictions, ratings.float())
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, movies, reinforcements, ratings = batch

        predictions = self(users, movies, reinforcements)
        val_loss = self.loss(predictions, ratings.float())
        score = get_score(predictions.cpu().numpy(), ratings.cpu().numpy())
        self.log('val_loss', val_loss)
        self.log('score', score)
        return val_loss

    def test_step(self, batch, batch_idx):
        users, movies, reinforcements = batch
        predictions = self(users, movies, reinforcements)
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

            # Initialize weights of the linear layers
            self.init_weights()

        def init_weights(self):
            nn.init.xavier_uniform_(self.transformation_layer_1.weight)
            nn.init.xavier_uniform_(self.transformation_layer_2.weight)

            self.transformation_layer_1.bias.data.fill_(0.01)
            self.transformation_layer_2.bias.data.fill_(0.01)

        def forward(self, previous_embedding, device):
            self.laplacian_matrix = self.laplacian_matrix.to(device)
            self.identity = self.identity.to(device)

            embedding_1 = torch.sparse.mm((self.laplacian_matrix + self.identity), previous_embedding)
            embedding_2 = torch.mul(torch.sparse.mm(self.laplacian_matrix, previous_embedding), previous_embedding)

            transformed_embedding_1 = self.transformation_layer_1(embedding_1)
            transformed_embedding_2 = self.transformation_layer_2(embedding_2)

            return nn.LeakyReLU()(transformed_embedding_1 + transformed_embedding_2)