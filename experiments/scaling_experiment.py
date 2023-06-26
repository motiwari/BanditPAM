import numpy as np
import os
import pandas as pd
from glob import glob

from run_all_versions import run_banditpam
from scripts.comparison_utils import print_results, store_results
from scripts.constants import (
    # Datasets
    MNIST,
    CIFAR,
)


def read_dataset(dataset_name):
    """
    Reads the specified dataset from the local storage.

    :param dataset_name: A string that represents the name of the dataset
    :return: The requested dataset as a numpy array
    """
    if dataset_name == MNIST:
        filename = "MNIST_70k"
        delimiter = " "
    elif dataset_name == CIFAR:
        filename = "cifar10"
        delimiter = ","

    dataset = pd.read_csv(
        os.path.join("data", f"{filename}.csv"), delimiter=delimiter, header=None
    ).to_numpy()
    return dataset


def scaling_experiment_with_k(
    dataset_name,
    n_medoids_list=None,
    algorithms=None,
    loss: str = "L2",
    verbose=True,
    save_logs=True,
    cache_width=1000,
    dirname="scaling_with_k",
    num_experiments=3,
):
    """
    Runs a scaling experiment varying the number of medoids (k),
    and stores the results in the appropriate log files.

    :param dataset_name: A string that represents the name of the dataset
    :param n_medoids_list: A list of integers specifying different number of medoids to run the experiment with
    :param algorithms: A list of strings specifying the names of algorithms to use in the experiment
    :param loss: A string specifying the type of loss to be used ("L2" is the default)
    :param verbose: A boolean indicating whether to print the results to the console
    :param save_logs: A boolean indicating whether to save the results to log files
    :param cache_width: An integer specifying the cache width for BanditPAM
    :param dirname: A string specifying the directory name where the log files will be saved
    :param num_experiments: An integer specifying the number of experiments to run
    """
    dataset = read_dataset(dataset_name)
    num_data = len(dataset)
    log_dir = os.path.join("logs", dirname)
    print("Running sampling complexity experiment with k on ", dataset_name)

    for experiment_index in range(num_experiments):
        print("\n\nExperiment: ", experiment_index)
        for n_medoids in n_medoids_list:
            print("\nNum medoids: ", n_medoids)
            for algorithm in algorithms:
                print("Running ", algorithm)
                log_name = (
                    f"{algorithm}_{dataset_name}_n{num_data}_idx{experiment_index}"
                )
                kmed, runtime = run_banditpam(
                    algorithm, dataset, n_medoids, loss, cache_width
                )

                if verbose:
                    print_results(kmed, runtime)

                if save_logs:
                    store_results(kmed, runtime, log_dir, log_name, num_data, n_medoids)


def scaling_experiment_with_n(
    dataset_name,
    num_data_list,
    n_medoids,
    algorithms=None,
    loss: str = "L2",
    verbose=True,
    save_logs=True,
    cache_width=1000,
    dirname="scaling_with_n",
    parallelize=True,
    num_experiments=3,
):
    """
    Runs a scaling experiment varying the number of data points (n),
    and stores the results in the appropriate log files.

    :param dataset_name: A string that represents the name of the dataset
    :param num_data_list: A list of integers specifying different number of data points to run the experiment with
    :param n_medoids: An integer specifying the number of medoids
    :param algorithms: A list of strings specifying the names of algorithms to use in the experiment
    :param loss: A string specifying the type of loss to be used ("L2" is the default)
    :param verbose: A boolean indicating whether to print the results to the console
    :param save_logs: A boolean indicating whether to save the results to log files
    :param cache_width: An integer specifying the cache width for BanditPAM
    :param dirname: A string specifying the directory name where the log files will be saved
    :param num_experiments: An integer specifying the number of experiments to run
    """
    dataset = read_dataset(dataset_name)
    log_dir = os.path.join("logs", dirname)

    print("Running sampling complexity experiment with n on ", dataset_name)

    for experiment_index in range(5, 5 + num_experiments):
        print("\n\nExperiment: ", experiment_index)
        for num_data in num_data_list:
            print("\nNum data: ", num_data)
            data_indices = np.random.randint(0, len(dataset), num_data)
            data = dataset[data_indices]
            for algorithm in algorithms:
                print("\n<Running ", algorithm, ">")
                log_name = (
                    f"{algorithm}_{dataset_name}_k{n_medoids}_idx{experiment_index}"
                )
                print(log_name)
                kmed, runtime = run_banditpam(
                    algorithm, data, n_medoids, loss, cache_width, parallelize
                )

                if verbose:
                    print_results(kmed, runtime)

                if save_logs:
                    store_results(kmed, runtime, log_dir, log_name, num_data, n_medoids)