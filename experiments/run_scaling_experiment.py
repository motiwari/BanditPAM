import numpy as np
import time

from scaling_experiment import scaling_experiment_with_k, scaling_experiment_with_n, debug_loss_with_n, \
    find_loss_mistmatch, run_sklearn, run_banditpam2
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

np.random.seed(1)


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
                                      dirname="complexity_debugging",
                                      num_experiments=2,
                                      )


def run_loss_debug_experiment():
    num_experiments = 1
    num_medoids = 10
    num_data_list = [10000]
    np.random.seed(5)
    start_time = time.time()
    sklearn_losses, sklearn_medoids = run_sklearn(dataset_name=MNIST, n_medoids=num_medoids,
                                                  num_data_list=num_data_list, n_swaps=0,
                                                  num_experiments=num_experiments)
    end_time = time.time()
    print("Time: ", end_time - start_time)
    print("n swap: ", 100)
    print("medoids: ", sklearn_medoids)

    for algorithm in [BANDITPAM_VA_CACHING, BANDITPAM_ORIGINAL_CACHING]:
        np.random.seed(5)
        banditpam_losses, banditpam_medoids = run_banditpam2(dataset_name=MNIST,
                                                             loss="L2",
                                                             algorithm=algorithm,
                                                             n_medoids=num_medoids,
                                                             num_data_list=num_data_list,
                                                             parallelize=False,
                                                             num_experiments=num_experiments,
                                                             save_logs=False,
                                                             n_swaps=0
                                                             )

        mismatch_indices = np.where(sklearn_losses != banditpam_losses)
        print("Num mistmatch:      ", len(mismatch_indices[0]))
        print("Mistmatches:        ", mismatch_indices)
        print("Sklearn Losses:     ", sklearn_losses)
        print("BanditPAM Losses:   ", banditpam_losses)
        print("Loss difference:    ", np.abs(np.sum(sklearn_losses) - np.sum(banditpam_losses)))

        for a, b in zip(sklearn_medoids, banditpam_medoids):
            print("Sklearn medoids:    ", a)
            print("BanditPAM medoids:  ", b)


if __name__ == "__main__":
    run_scaling_experiment_with_n()
    # run_loss_debug_experiment()
