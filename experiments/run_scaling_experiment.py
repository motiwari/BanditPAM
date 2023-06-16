import numpy as np

from scaling_experiment import scaling_experiment_with_k, scaling_experiment_with_n, sklearn_speed_experiment
from scripts.constants import (
    # Algorithms
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_CACHING,
    BANDITPAM_VA_NO_CACHING,
    ALL_BANDITPAMS,

    # Datasets
    MNIST,
    CIFAR,
)


def get_loss_function(dataset):
    """
    Returns the appropriate loss function based on the dataset.

    :param dataset: A string that represents the name of the dataset
    :return: A string indicating the type of loss function ("L1" or "L2")
    """
    if dataset == MNIST:
        return "L1"
    else:
        return "L2"


def get_num_data_list(dataset):
    """
    Returns a list of numbers indicating the different number of data points to run the experiment with,
    based on the dataset.

    :param dataset: A string that represents the name of the dataset
    :return: A numpy array specifying different numbers of data points
    """
    if dataset == MNIST:
        num_data = 70000
        return np.linspace(10000, num_data, 5, dtype=int)
    elif dataset == CIFAR:
        num_data = 40000
        return np.linspace(10000, num_data, 5, dtype=int)


def run_scaling_experiment_with_k():
    """
    Runs scaling experiments varying the number of medoids (k) for the MNIST and CIFAR datasets
    using all BanditPAM algorithms.
    """
    for dataset in [MNIST, CIFAR]:
        loss = get_loss_function(dataset)
        scaling_experiment_with_k(dataset_name=dataset,
                                  loss=loss,
                                  algorithms=ALL_BANDITPAMS,
                                  n_medoids_list=[5, 8, 10]
                                  )


def run_scaling_experiment_with_n():
    """
    Runs scaling experiments varying the number of data points (n) for the MNIST and CIFAR datasets
    using all BanditPAM algorithms.
    """
    for dataset in [MNIST]:
        loss = get_loss_function(dataset)
        num_data_list = get_num_data_list(dataset)
        for n_medoids in [10]:
            scaling_experiment_with_n(dataset_name=dataset,
                                      loss=loss,
                                      algorithms=ALL_BANDITPAMS,
                                      n_medoids=n_medoids,
                                      num_data_list=num_data_list,
                                      dirname="scaling_with_n",
                                      num_experiments=3, )


def run_sklearn_speed_experiment():
    sklearn_speed_experiment()


if __name__ == "__main__":
    run_sklearn_speed_experiment()