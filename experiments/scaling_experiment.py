import numpy as np
import os
import pandas as pd

from run_all_versions import run_banditpam
from scripts.comparison_utils import print_results, store_results
from scripts.constants import (
    # Algorithms
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_NO_CACHING,
    BANDITPAM_VA_CACHING,
    ALL_BANDITPAMS,

    # Datasets
    MNIST,
    SCRNA,
    CIFAR,
)


def read_dataset(dataset_name):
    if dataset_name == MNIST:
        filename = "MNIST_70k"
        delimiter = " "
    elif dataset_name == CIFAR:
        filename = "cifar10"
        delimiter = ","

    dataset = pd.read_csv(os.path.join("data", f"{filename}.csv"), delimiter=delimiter, header=None).to_numpy()
    return dataset


def scaling_experiment_with_k(dataset_name,
                              n_medoids_list=None,
                              algorithms=None,
                              loss: str = "L2",
                              verbose=True,
                              save_logs=True,
                              cache_width=1000,
                              dirname="scaling_with_k_cluster",
                              num_experiments=3):
    dataset = read_dataset(dataset_name)
    num_data = len(dataset)
    log_dir = os.path.join("scripts", "experiments", "logs", dirname)
    print("Running sampling complexity experiment with k on ", dataset_name)

    for experiment_index in range(num_experiments):
        print("Experiment: ", experiment_index)
        for n_medoids in n_medoids_list:
            print("Num medoids: ", n_medoids)
            for algorithm in algorithms:
                print("Running ", algorithm)
                log_name = f"{algorithm}_{dataset_name}_n{num_data}_idx{experiment_index}"
                kmed, runtime = run_banditpam(algorithm, dataset, n_medoids, loss, cache_width)

                if verbose:
                    print_results(kmed, runtime)

                if save_logs:
                    store_results(kmed, runtime, log_dir, log_name, num_data, n_medoids)

                print("\n" * 2)

            print("\n" * 3)


def scaling_experiment_with_n(dataset_name,
                              num_data_list,
                              n_medoids,
                              algorithms=None,
                              loss: str = "L2",
                              verbose=True,
                              save_logs=True,
                              cache_width=1000,
                              dirname="scaling_with_n_cluster",
                              num_experiments=3):
    dataset = read_dataset(dataset_name)
    log_dir = os.path.join("scripts", "experiments", "logs", dirname)

    print("Running sampling complexity experiment with n on ", dataset_name)

    for experiment_index in range(num_experiments):
        print("Experiment: ", experiment_index)
        for num_data in num_data_list:
            print("Num data: ", num_data)
            data_indices = np.random.randint(0, len(dataset), num_data)
            data = dataset[data_indices]

            for algorithm in algorithms:
                print("Running ", algorithm)
                log_name = f"{algorithm}_{dataset_name}_k{n_medoids}_idx{experiment_index}"
                kmed, runtime = run_banditpam(algorithm, data, n_medoids, loss, cache_width)

                if verbose:
                    print_results(kmed, runtime)

                if save_logs:
                    store_results(kmed, runtime, log_dir, log_name, num_data, n_medoids)

                print("\n" * 2)

            print("\n" * 3)
