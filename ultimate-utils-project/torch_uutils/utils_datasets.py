import argparse
import logging

DATASETS = ["CIFAR10", "CIFAR100", "ImageNet"]


def parse_args():
    """
    Parses the arguments passed in.
    """
    parser = argparse.ArgumentParser(description="archMAML")
    parser.add_argument(
        "--nodes",
        metavar="B",
        type=int,
        help="number of nodes per cell",
        default=5
    )
    parser.add_argument(
        "--layers", metavar="N", type=int, help="number of layers", default=3
    )
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="D",
        type=str,
        choices=DATASETS,
        help="datasets: [" + ", ".join(DATASETS) + "]",
        required=True,
    )
    decoder_args = parser.add_argument_group("Decoder Arguments:")
    decoder_args.add_argument(
        "--decoder-layers",
        type=int,
        help="number of layers for the decoder.",
        default=1
    )
    decoder_args.add_argument(
        "--decoder-hidden-size",
        type=int,
        help="size of hidden state of decoder."
    )
    decoder_args.add_argument(
        "--decoder-dropout",
        type=float,
        help="dropout probability of the decoder."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    return args


def get_logger(log_path, log_filename):
    """
    Returns logger.

    Args:
        log_path: the path at which the logs will be stored
        log_filename: filename of the log

    Returns:
        logger
    """
    logger = logging.getLogger(log_filename)
    file_handler = logging.FileHandler(log_filename + ".log")
    logger.addHandler(file_handler)

    return logger
