from comet_ml import Experiment
from modules import (
    test_reliability_predictor_old,
    test_old
)
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utilities.helper import (
    create_argument_parser,
    create_comet_logger,
    get_config,
    free_gpu_memory
)
from utilities.data_preparation import (
    load_data,
    load_reliabilities,
    create_dataset,
    create_surprise_data,
    create_dataset_with_reliabilities,
    create_laplacian_matrix
)
from utilities.evaluation_functions import get_reliability
from surprise import SVDpp
from surprise.model_selection import cross_validate


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    config = get_config(args, test_old.hyper_parameters)

    pl.seed_everything(args.random_seed)
    np.random.seed(7)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(config)

    train_pd = load_data(
        file_path="cache_old/train_data.csv",
        full_dataset=args.leonhard,
        train_val_split=False
    )
    val_pd = load_data(
        file_path="cache_old/val_data.csv",
        full_dataset=args.leonhard,
        train_val_split=False
    )
    test_pd = load_data(
        file_path="cache_old/test_data.csv",
        full_dataset=args.leonhard,
        train_val_split=False
    )

    train_reliabilities = load_reliabilities("cache_old/train_reliabilities.csv")
    val_reliabilities = load_reliabilities("cache_old/val_reliabilities.csv")
    test_reliabilities = load_reliabilities("cache_old/test_reliabilities.csv")

    train_data = create_dataset_with_reliabilities(train_pd, train_reliabilities)
    val_data = create_dataset_with_reliabilities(val_pd, val_reliabilities)
    test_ids, test_data = create_dataset_with_reliabilities(test_pd, test_reliabilities, test_dataset=True)

    laplacian_matrix = create_laplacian_matrix(train_pd, config['number_of_users'], config['number_of_movies'])

    graph_neural_network = test_old.GNN(train_data, val_data, test_data, test_ids, args, laplacian_matrix, config)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['patience']
    )
    trainer = pl.Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                         max_epochs=config['num_epochs'],
                         logger=comet_logger,
                         callbacks=[early_stopping])

    trainer.fit(graph_neural_network)
    trainer.test(graph_neural_network)


if __name__ == "__main__":
    main()
