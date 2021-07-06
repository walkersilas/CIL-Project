from comet_ml import Experiment
from modules import ncf_baseline
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from utilities.helper import (
    create_argument_parser,
    create_comet_logger,
    get_config,
    get_hash
)
from utilities.data_preparation import (
    load_data,
    create_dataset
)


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    config = get_config(args, ncf_baseline.hyper_parameters)

    pl.seed_everything(args.random_seed)
    np.random.seed(7)

    comet_logger = create_comet_logger(args)
    comet_logger.log_hyperparams(config)

    train_pd, val_pd = load_data(
        file_path=args.data_dir + args.train_data,
        train_val_split=True,
        random_seed=args.random_seed,
        train_size=config['train_size']
    )
    test_pd = load_data(
        file_path=args.data_dir + args.test_data,
        train_val_split=False
    )
    train_data, val_data = create_dataset(train_pd), create_dataset(val_pd)
    test_ids, test_data = create_dataset(test_pd, test_dataset=True)

    ncf = ncf_baseline.NCF(train_data, val_data, test_data, test_ids, args, config)

    checkpoint_filename = "ncf_baseline_" + str(get_hash(config, args))
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=checkpoint_filename,
        monitor='val_loss',
        save_top_k=1,
        mode="min"
    )
    trainer = pl.Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                         max_epochs=1, #config['num_epochs'],
                         logger=comet_logger,
                         callbacks=[checkpoint_callback])

    trainer.fit(ncf)

    best_ncf = ncf_baseline.NCF.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                                            train_data=train_data,
                                                                            val_data=val_data,
                                                                            test_data=test_data,
                                                                            test_ids=test_ids,
                                                                            args=args,
                                                                            config=config)

    trainer.test(best_ncf)


if __name__ == "__main__":
    main()
