from comet_ml import Experiment
import json
from modules import gnn_baseline
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

    # Update config with parameters specified in config.json
    if args.config is not None:
        try:
            new_config = json.load(open(args.config))

            for key in new_config.keys():
                gnn_baseline.hyper_parameters[key] = new_config[key]

        except:
            print("New config not found ... Continue with default config of model ...")

    pl.seed_everything(args.random_seed)
    np.random.seed(7)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(gnn_baseline.hyper_parameters)

    train_pd, val_pd = load_data(
        file_path=args.data_dir + args.train_data,
        full_dataset=args.leonhard,
        train_val_split=True,
        random_seed=args.random_seed,
        train_size=gnn_baseline.hyper_parameters['train_size']
    )
    test_pd = load_data(
        file_path=args.data_dir + args.test_data,
        full_dataset=args.leonhard,
        train_val_split=False
    )
    train_data, val_data = create_dataset(train_pd), create_dataset(val_pd)
    test_ids, test_data = create_dataset(test_pd, test_dataset=True)

    laplacian_matrix = create_laplacian_matrix(train_pd,
                            gnn_baseline.hyper_parameters['number_of_users'],
                            gnn_baseline.hyper_parameters['number_of_movies']
    )

    graph_neural_network = gnn_baseline.GNN(train_data, val_data, test_data, test_ids, args, laplacian_matrix)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=gnn_baseline.hyper_parameters['patience']
    )
    trainer = pl.Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                         max_epochs=gnn_baseline.hyper_parameters['num_epochs'],
                         logger=comet_logger,
                         callbacks=[early_stopping])

    trainer.fit(graph_neural_network)
    trainer.test(graph_neural_network)

if __name__ == "__main__":
    main()
