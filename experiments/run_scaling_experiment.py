import numpy as np

from scaling_experiment import (
    scaling_experiment_with_k,
    scaling_experiment_with_n,
)
from scripts.constants import (
    # Algorithms
    ALL_BANDITPAMS,
    # Datasets
    MNIST,
    CIFAR,
    SCRNA,
)


def get_loss_function(dataset):
    """
    Returns the appropriate loss function based on the dataset.

    :param dataset: A string that represents the name of the dataset
    :return: A string indicating the type of loss function ("L1" or "L2")
    """
    if dataset in [MNIST, SCRNA]:
        return "L1"
    else:
        return "L2"


def get_num_data_list(dataset):
    """
    Returns a list of numbers indicating the different number of data points to
    run the experiment with, based on the dataset.

    :param dataset: A string that represents the name of the dataset
    :return: A numpy array specifying different numbers of data points
    """
    if dataset == MNIST:
        num_data = 70000
    else:
        num_data = 40000

    return np.linspace(10000, num_data, 4, dtype=int)


def run_scaling_experiment_with_k():
    """
    Runs scaling experiments varying the number of medoids (k) for the MNIST
    and CIFAR datasets using all BanditPAM algorithms.
    """
    for dataset in [MNIST, CIFAR]:
        loss = get_loss_function(dataset)
        scaling_experiment_with_k(
            dataset_name=dataset,
            loss=loss,
            algorithms=ALL_BANDITPAMS,
            n_medoids_list=[5, 8, 10],
        )


def run_scaling_experiment_with_n():
    """
    Runs scaling experiments varying the number of data points (n) for the
    MNIST and CIFAR datasets using all BanditPAM algorithms.
    """
    for dataset in [SCRNA]:
        loss = get_loss_function(dataset)
        num_data_list = get_num_data_list(dataset)
        for n_medoids in [5]:
            np.random.seed(0)
            scaling_experiment_with_n(
                dataset_name=dataset,
                loss=loss,
                algorithms=ALL_BANDITPAMS,
                n_medoids=n_medoids,
                num_data_list=num_data_list,
                dirname="scaling_with_n",
                num_experiments=10,
            )


if __name__ == "__main__":
    run_scaling_experiment_with_n()
