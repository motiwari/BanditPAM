"""
Convenience functions to retrieve command-line arguments and load datasets.
"""

import numpy as np
import mnist
import argparse
from keras.datasets import cifar10
import ssl
import cv2

# for downloading CIFAR dataset
ssl._create_default_https_context = ssl._create_unverified_context


def get_args(arguments):
    """
    Retrieve command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-k", "--num_medoids", help="Number of medoids", type=int, default=3
    )
    parser.add_argument(
        "-N",
        "--sample_size",
        help="Sampling size of dataset",
        type=int,
        default=700,
    )
    parser.add_argument(
        "-s", "--seed", help="Random seed", type=int, default=42
    )
    parser.add_argument(
        "-d", "--dataset", help="Dataset to use", type=str, default="MNIST"
    )
    parser.add_argument(
        "-m", "--metric", help="Metric to use (L1 or L2)", type=str
    )
    parser.add_argument(
        "-e",
        "--exp_config",
        help="Experiment configuration file to use",
        required=False,
    )
    args = parser.parse_args(arguments)
    return args


def load_data(args):
    """
    Load the different datasets as a numpy matrix.
    """
    if args.dataset == "MNIST":
        N = 70000
        m = 28
        train_images = mnist.train_images()
        test_images = mnist.test_images()

        total_images = np.append(train_images, test_images, axis=0)

        assert (total_images == np.vstack((train_images, test_images))).all()
        assert total_images.shape == (N, m, m)

        # Normalizing images
        return total_images.reshape(N, m * m) / 255
    elif args.dataset == "SCRNA":
        file = "../data/scrna_reformat.csv"
        data_ = np.loadtxt(file, delimiter=",")
        return data_
    elif args.dataset == "CIFAR":
        N = 60000
        m = 32
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        total_images = np.append(X_train, X_test, axis=0)
        # convert images to grayscale
        total_images = np.array(
            [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in total_images]
        )

        assert total_images.shape == (N, m, m)

        total_images = total_images.astype("float32")
        return total_images.reshape(N, m * m) / 255
    else:
        raise Exception("Didn't specify a valid dataset")
