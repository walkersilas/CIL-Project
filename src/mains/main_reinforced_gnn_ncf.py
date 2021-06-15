from comet_ml import Experiment
from modules import reinforced_gnn_ncf
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utilities.helper import (
    create_argument_parser,
    create_comet_logger,
    get_config
)
from utilities.data_preparation import (
    load_data,
    load_reinforcements,
    create_dataset,
    create_surprise_data,
    create_dataset_with_reinforcements,
    create_laplacian_matrix
)

def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    config = get_config(args, reinforced_gnn_ncf.hyper_parameters)

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

    reinforcement_cache = "cache/" + config["reinforcement_type"] + "/"
    train_reinforcements = load_reinforcements(reinforcement_cache + "train_reinforcement.csv")
    val_reinforcements = load_reinforcements(reinforcement_cache + "val_reinforcement.csv")
    test_reinforcements = load_reinforcements(reinforcement_cache + "test_reinforcement.csv")

    train_data = create_dataset_with_reinforcements(train_pd, train_reinforcements)
    val_data = create_dataset_with_reinforcements(val_pd, val_reinforcements)
    test_ids, test_data = create_dataset_with_reinforcements(test_pd, test_reinforcements, test_dataset=True)

    laplacian_matrix = create_laplacian_matrix(train_pd, config['number_of_users'], config['number_of_movies'])

    graph_neural_network = reinforced_gnn_ncf.GNN(train_data, val_data, test_data, test_ids, args, laplacian_matrix, config)
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
