from comet_ml import Experiment
from modules import (
    test_reliability_predictor,
    test
)
import numpy as np
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

    config = get_config(args, test.hyper_parameters)

    pl.seed_everything(args.random_seed)
    np.random.seed(7)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(config)

    train_pd = load_data(
        file_path="cache/train_data.csv",
        full_dataset=args.leonhard,
        train_val_split=False
    )
    val_pd = load_data(
        file_path="cache/val_data.csv",
        full_dataset=args.leonhard,
        train_val_split=False
    )
    test_pd = load_data(
        file_path="cache/test_data.csv",
        full_dataset=args.leonhard,
        train_val_split=False
    )

    train_raliabilities = load_reliabilities("cache/train_reliabilities.csv")
    val_reliabilities = load_reliabilities("cache/val_reliabilities.csv")
    test_reliabilities = load_reliabilities("cache/test_reliabilities.csv")

    train_data = create_dataset_with_reliabilities(train_pd, train_raliabilities)
    val_data = create_dataset_with_reliabilities(val_pd, val_reliabilities)
    test_ids, test_data = create_dataset_with_reliabilities(test_pd, test_reliabilities, test_dataset=True)

    graph_neural_network = gnn_ncf.GNN(train_data, val_data, test_data, test_ids, args, laplacian_matrix, config)
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
