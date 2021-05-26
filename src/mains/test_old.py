from comet_ml import Experiment
from modules import test
import numpy as np
import torch
import pytorch_lightning as pl
from utilities.helper import (
    create_argument_parser,
    create_comet_logger
)
from utilities.data_preparation import (
    load_data,
    create_dataset,
    create_surprise_data
)

from surprise import SVDpp

from utilities.evaluation_functions import get_score


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    pl.seed_everything(args.random_seed)
    np.random.seed(7)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(test.hyper_parameters)

    train_pd, val_pd = load_data(
        file_path=args.data_dir + args.train_data,
        full_dataset=args.leonhard,
        train_val_split=True,
        random_seed=args.random_seed,
        train_size=test.hyper_parameters['train_size']
    )
    test_pd = load_data(
        file_path=args.data_dir + args.test_data,
        full_dataset=args.leonhard,
        train_val_split=False
    )
    train_data, val_data = create_dataset(train_pd), create_dataset(val_pd)
    test_ids, test_data = create_dataset(test_pd, test_dataset=True)

    surprise_train_data = create_surprise_data(train_pd).build_full_trainset()

    svd_pp = SVDpp(n_factors=12, lr_all=0.085, n_epochs=50, reg_all=0.01, verbose=True)
    svd_pp.fit(surprise_train_data)

    ncf = test.NCF(train_data, val_data, test_data, test_ids, args, svd_pp)
    trainer = pl.Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                         max_epochs=test.hyper_parameters['num_epochs'],
                         logger=comet_logger)

    trainer.fit(ncf)
    trainer.test(ncf)


if __name__ == "__main__":
    main()
