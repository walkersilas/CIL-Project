from comet_ml import Experiment
import numpy as np
import pandas as pd
import os
import torch
import pytorch_lightning as pl
from utilities.helper import (
    create_argument_parser,
    create_comet_logger,
    get_config,
    free_gpu_memory
)
from utilities.data_preparation import (
    load_data,
    create_dataset,
    create_surprise_data,
    create_surprise_data_without_val,
    create_dataset_with_reliabilities
)
from utilities.evaluation_functions import get_reliability
from surprise import KNNBaseline
from surprise.model_selection import cross_validate


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


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    config = get_config(args, hyper_parameters)

    pl.seed_everything(args.random_seed)
    np.random.seed(7)

    comet_logger = create_comet_logger(args, prefix="reliability_predictor")
    comet_logger.log_hyperparams(config)

    train_pd, val_pd = load_data(
        file_path=args.data_dir + args.train_data,
        full_dataset=args.leonhard,
        train_val_split=True,
        random_seed=args.random_seed,
        train_size=config['train_size']
    )
    test_pd = load_data(
        file_path=args.data_dir + args.test_data,
        full_dataset=args.leonhard,
        train_val_split=False
    )
    if not os.path.exists("cache"):
        os.mkdir("cache")

    train_pd.to_csv("cache/train_data.csv", index=False)
    val_pd.to_csv("cache/val_data.csv", index=False)
    test_pd.to_csv("cache/test_data.csv", index=False)


    train_data, val_data = create_dataset(train_pd), create_dataset(val_pd)
    test_ids, test_data = create_dataset(test_pd, test_dataset=True)
    _, val_data_no_labels = create_dataset(val_pd, test_dataset=True)

    surprise_train_data = create_surprise_data_without_val(train_pd).build_full_trainset()

    knn = KNNBaseline(k=99, min_k=11, sim_options={"name": "pearson_baseline", "user_based": False}, bsl_options={"method": "als"})
    knn.fit(surprise_train_data)


    train_reliabilities = []
    for user, movie, rating in train_data:
        prediction = knn.predict(user.item(), movie.item()).est
        train_reliabilities.append(prediction)

    train_reliabilities_pd = pd.DataFrame({
        "Reliability": train_reliabilities
    })
    train_reliabilities_pd.to_csv("cache/train_reliabilities.csv", index=False)

    val_reliabilities = []
    for user, movie, rating in val_data:
        prediction = knn.predict(user.item(), movie.item()).est
        val_reliabilities.append(prediction)

    val_reliabilities_pd = pd.DataFrame({
        "Reliability": val_reliabilities
    })
    val_reliabilities_pd.to_csv("cache/val_reliabilities.csv", index=False)

    test_reliabilities = []
    for user, movie in test_data:
        prediction = knn.predict(user.item(), movie.item()).est
        test_reliabilities.append(prediction)

    test_reliabilities_pd = pd.DataFrame({
        "Reliability": test_reliabilities
    })
    test_reliabilities_pd.to_csv("cache/test_reliabilities.csv", index=False)


if __name__ == "__main__":
    main()
