from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace
)
import json
from pytorch_lightning.loggers import CometLogger


def create_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Main entry point for this project",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../data/",
        help="path to the directory containing the unprocessed data"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data_train.csv",
        help="name of the training data file"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data_test.csv",
        help="name of the testing data file"
    )
    parser.add_argument(
        "--leonhard",
        action="store_true",
        help="flag indicating whether the model is run in the leonhard cluster"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="random seed used"
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="flag indicating whether the experiment is logged in comet ml"
    )
    parser.add_argument(
        "--comet-key",
        type=str,
        default="../../comet.json",
        help="path to the comet api key directory"
    )
    parser.add_argument(
        "--comet-directory",
        type=str,
        default="./logs",
        help="path to log directory when comet can not be run online"
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="number of dataloader workers used"
    )
    return parser


def create_comet_logger(args: Namespace) -> CometLogger:
    comet_api_key = None
    try:
        comet_api_key = json.load(open(args.comet_key))
    except:
        print("Comet API Key not found ... Continue by logging the experiment offline ...")

    if comet_api_key is None:
        return CometLogger(
            save_dir=args.comet_directory
        )
    else:
        return CometLogger(
            api_key=comet_api_key["api_key"],
            project_name=comet_api_key["project_name"],
            workspace=comet_api_key["workspace"],
            disabled=args.disable_logging
        )