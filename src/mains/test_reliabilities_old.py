from comet_ml import Experiment
from modules import test_reliability_predictor_old
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
    create_dataset_with_reliabilities
)
from utilities.evaluation_functions import get_reliability
from surprise import SVDpp
from surprise.model_selection import cross_validate


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    config = get_config(args, test_reliability_predictor_old.hyper_parameters)

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
    if not os.path.exists("cache_old"):
        os.mkdir("cache_old")

    train_pd.to_csv("cache_old/train_data.csv", index=False)
    val_pd.to_csv("cache_old/val_data.csv", index=False)
    test_pd.to_csv("cache_old/test_data.csv", index=False)


    train_data, val_data = create_dataset(train_pd), create_dataset(val_pd)
    test_ids, test_data = create_dataset(test_pd, test_dataset=True)
    _, val_data_no_labels = create_dataset(val_pd, test_dataset=True)

    surprise_train_data = create_surprise_data(train_pd, val_pd).build_full_trainset()

    svd_pp = SVDpp(n_factors=12, lr_all=0.0001, n_epochs=10, reg_all=0.01, verbose=True)
    svd_pp.fit(surprise_train_data)


    train_reliabilities = []
    for user, movie, rating in train_data:
        prediction = svd_pp.predict(user.item(), movie.item()).est
        train_reliabilities.append(get_reliability(prediction, rating))

    train_reliabilities_pd = pd.DataFrame({
        "Reliability": train_reliabilities
    })
    train_reliabilities_pd.to_csv("cache_old/train_reliabilities.csv", index=False)

    val_reliabilities = []
    for user, movie, rating in val_data:
        prediction = svd_pp.predict(user.item(), movie.item()).est
        val_reliabilities.append(get_reliability(prediction, rating))

    train_data = create_dataset_with_reliabilities(train_pd, train_reliabilities)
    val_data = create_dataset_with_reliabilities(val_pd, val_reliabilities)


    reliability_predictor = test_reliability_predictor_old.RELIABILITY_PREDICTOR(
        train_data, val_data, val_data_no_labels, test_data, args, config
    )
    trainer = pl.Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                         max_epochs=config['num_epochs'],
                         logger=comet_logger)

    trainer.fit(reliability_predictor)
    trainer.test(reliability_predictor)


if __name__ == "__main__":
    main()
