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
from tqdm import tqdm
torch.manual_seed(7)
np.random.seed(7)

comet_api_key_path = "../../../API_Keys/comet.json"
comet_api_key = json.load(open(comet_api_key_path))
experiment = Experiment(
    api_key=comet_api_key["api_key"],
    project_name=comet_api_key["project_name"],
    workspace=comet_api_key["workspace"]
)


class NCF(nn.Module):
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

    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))


train_data_path = '../data/data_train.csv'
test_data_path = '../data/data_test.csv'

number_of_users, number_of_movies = (10000, 1000)
train_size = 0.9

reduce_dataset = False

hyper_params = {
    'batch_size': 1024,
    'num_epochs': 25,
    'embedding_size': 256,
    'learning_rate': 1e-3,
}
experiment.log_parameters(hyper_params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


def rmse(x, y):
    return math.sqrt(mean_squared_error(x, y))


def get_score(predictions, target_values):
    return rmse(predictions, target_values)


def load_data(file_path):
    data_pd = pd.read_csv(file_path)

    if reduce_dataset:
        data_pd = data_pd.head(10000)

    print(data_pd.head(5))
    print()
    print('Shape', data_pd.shape)

    train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
    return train_pd, test_pd


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def create_dataloader(users, movies, predictions):
    users_torch = torch.tensor(users, device=device, dtype=torch.int64)
    movies_torch = torch.tensor(movies, device=device, dtype=torch.int64)
    predictions_torch = torch.tensor(predictions, device=device, dtype=torch.int64)

    dataloader = DataLoader(
        TensorDataset(users_torch, movies_torch, predictions_torch),
        batch_size=hyper_params['batch_size']
    )

    return dataloader


def train_network(ncf, optimizer, train_dataloader, test_dataloader, test_predictions):
    with tqdm(total=len(train_dataloader) * hyper_params['num_epochs']) as pbar:
        for epoch in range(hyper_params['num_epochs']):
            with experiment.train():
                for users_batch, movies_batch, target_predictions_batch in train_dataloader:
                    optimizer.zero_grad()

                    predictions_batch = ncf(users_batch, movies_batch)
                    loss = mse_loss(predictions_batch, target_predictions_batch)
                    loss.backward()
                    optimizer.step()

                    pbar.update(1)

            with experiment.test():
                with torch.no_grad():
                    all_predictions = []
                    for users_batch, movies_batch, _ in test_dataloader:
                        predictions_batch = ncf(users_batch, movies_batch)
                        all_predictions.append(predictions_batch)

                all_predictions = torch.cat(all_predictions)

                reconstruction_rmse = get_score(all_predictions.cpu().numpy(), test_predictions)
                pbar.set_description('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstruction_rmse))
                experiment.log_metric("Reconstruction RMSE", reconstruction_rmse, step=epoch)
    return ncf


def main():
    train_pd, test_pd = load_data(train_data_path)
    train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
    test_users, test_movies, test_predictions = extract_users_items_predictions(test_pd)

    ncf = NCF(number_of_users, number_of_movies, hyper_params['embedding_size']).to(device)
    optimizer = optim.Adam(ncf.parameters(), lr=hyper_params['learning_rate'])

    train_dataloader = create_dataloader(train_users, train_movies, train_predictions)
    test_dataloader = create_dataloader(test_users, test_movies, test_predictions)

    ncf = train_network(ncf, optimizer, train_dataloader, test_dataloader, test_predictions)


if __name__ == '__main__':
    main()
