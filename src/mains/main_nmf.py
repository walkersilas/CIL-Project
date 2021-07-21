"""NMF main.

This modules trains and tests the NMF model.
"""

import numpy as np
from comet_ml import Experiment

from modules import nmf
from utilities.data_preparation import create_dataset, create_surprise_data, load_data
from utilities.helper import create_argument_parser, create_comet_logger, get_config


def main():
    """Run NMF model."""
    parser = create_argument_parser()
    args = parser.parse_args()

    config = get_config(args, nmf.hyper_parameters)

    np.random.seed(7)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(config)

    train_pd = load_data(file_path=args.data_dir + args.train_data, train_val_split=False)
    test_pd = load_data(file_path=args.data_dir + args.test_data, train_val_split=False)

    train_data = create_surprise_data(train_pd).build_full_trainset()
    test_ids, test_data = create_dataset(test_pd, test_dataset=True)

    algo = nmf.NMF(train_data, test_data, test_ids, args, config, comet_logger)

    algo.fit()
    algo.test()


if __name__ == "__main__":
    main()
