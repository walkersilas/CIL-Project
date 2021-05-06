from comet_ml import Experiment
from modules import gnn
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utilities.helper import (
    create_argument_parser,
    create_comet_logger
)
from utilities.data_preparation import (
    load_data,
    create_dataset,
    create_laplacian_matrix
)


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    pl.seed_everything(args.random_seed)
    np.random.seed(7)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(gnn.hyper_parameters)

    train_pd, val_pd = load_data(
        file_path=args.data_dir + args.train_data,
        full_dataset=args.leonhard,
        train_val_split=True,
        random_seed=args.random_seed,
        train_size=gnn.hyper_parameters['train_size']
    )
    test_pd = load_data(
        file_path=args.data_dir + args.test_data,
        full_dataset=args.leonhard,
        train_val_split=False
    )
    train_data, val_data = create_dataset(train_pd), create_dataset(val_pd)
    test_ids, test_data = create_dataset(test_pd, test_dataset=True)

    laplacian_matrix = create_laplacian_matrix(train_pd,
                            gnn.hyper_parameters['number_of_users'],
                            gnn.hyper_parameters['number_of_movies']
    )

    graph_neural_network = gnn.GNN(train_data, val_data, test_data, test_ids, args, laplacian_matrix)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=gnn.hyper_parameters['patience']
    )
    trainer = pl.Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                         max_epochs=gnn.hyper_parameters['num_epochs'],
                         logger=comet_logger,
                         callbacks=[early_stopping])

    trainer.fit(graph_neural_network)
    trainer.test(graph_neural_network)


if __name__ == "__main__":
    main()
