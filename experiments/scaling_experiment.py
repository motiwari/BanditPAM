import numpy as np
import os
import pandas as pd
from sklearn_extra.cluster import KMedoids
from scipy.spatial import distance_matrix
from tqdm import tqdm

from run_all_versions import run_banditpam
from scripts.comparison_utils import print_results, store_results
from scripts.constants import (
    # Datasets
    MNIST,
    CIFAR,
    BANDITPAM_ORIGINAL_NO_CACHING,
)

np.random.seed(3)


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

    dataset = pd.read_csv(os.path.join("data", f"{filename}.csv"), delimiter=delimiter, header=None).to_numpy()
    return dataset


def scaling_experiment_with_k(dataset_name,
                              n_medoids_list=None,
                              algorithms=None,
                              loss: str = "L2",
                              verbose=True,
                              save_logs=True,
                              cache_width=1000,
                              dirname="scaling_with_k",
                              num_experiments=3):
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
                log_name = f"{algorithm}_{dataset_name}_n{num_data}_idx{experiment_index}"
                kmed, runtime = run_banditpam(algorithm, dataset, n_medoids, loss, cache_width)

                if verbose:
                    print_results(kmed, runtime)

                if save_logs:
                    store_results(kmed, runtime, log_dir, log_name, num_data, n_medoids)


def scaling_experiment_with_n(dataset_name,
                              num_data_list,
                              n_medoids,
                              algorithms=None,
                              loss: str = "L2",
                              verbose=True,
                              save_logs=True,
                              cache_width=1000,
                              dirname="scaling_with_n",
                              parallelize=True,
                              num_experiments=3):
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

    for experiment_index in range(3, 3+num_experiments):
        print("\n\nExperiment: ", experiment_index)
        for num_data in num_data_list:
            print("\nNum data: ", num_data)
            data_indices = np.random.randint(0, len(dataset), num_data)
            data = dataset[data_indices]
            for algorithm in algorithms:
                if algorithm is BANDITPAM_ORIGINAL_NO_CACHING and num_data == 55000: continue
                print("\n<Running ", algorithm, ">")
                log_name = f"{algorithm}_{dataset_name}_k{n_medoids}_idx{experiment_index}"
                kmed, runtime = run_banditpam(algorithm, data, n_medoids, loss, cache_width, parallelize)

                if verbose:
                    print_results(kmed, runtime)

                if save_logs:
                    store_results(kmed, runtime, log_dir, log_name, num_data, n_medoids)


def run_sklearn(dataset_name, num_data_list, n_medoids, n_swaps, num_experiments=3):
    print("Running SKLEARN")
    dataset = read_dataset(dataset_name)
    losses = []
    medoids = []

    for experiment_index in tqdm(range(num_experiments), desc="Experiment"):
        for num_data in tqdm(num_data_list, desc="Num data", leave=False):
            data_indices = np.random.randint(0, len(dataset), num_data)
            if experiment_index == 0: print(data_indices[0])
            data = dataset[data_indices]

            kmedoids = KMedoids(n_clusters=n_medoids, method="pam", init="build", max_iter=n_swaps).fit(data)
            sklearn_build_medoids = data[kmedoids.medoid_indices_, :]

            sklearn_medoids_ref_cost_distance_matrix = distance_matrix(sklearn_build_medoids, data)
            sklearn_objective = np.sum(np.min(sklearn_medoids_ref_cost_distance_matrix, 0))
            losses += sklearn_objective,
            medoids += kmedoids.medoid_indices_,

    return np.array(losses), medoids


def run_banditpam2(dataset_name,
                   num_data_list,
                   n_medoids,
                   algorithm=None,
                   loss: str = "L2",
                   verbose=True,
                   save_logs=True,
                   cache_width=1000,
                   dirname="scaling_with_n",
                   parallelize=False,
                   build_confidence=5,
                   swap_confidene=5,
                   n_swaps=0,
                   num_experiments=3):
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

    # exp 1, seed 5: original is worse than va and sklearn
    dataset = read_dataset(dataset_name)
    print("Running BanditPAM ", dataset_name, " Swap confidence: ", swap_confidene)
    losses = []
    medoids = []

    for experiment_index in tqdm(range(num_experiments), desc="Experiment"):
        for num_data in tqdm(num_data_list, desc="Num data", leave=False):
            data_indices = np.random.randint(0, len(dataset), num_data)
            if experiment_index == 0: print(data_indices[0])
            data = dataset[data_indices]

            kmed, runtime = run_banditpam(algorithm, data, n_medoids, loss, cache_width, parallelize, n_swaps,
                                          build_confidence, swap_confidene)
            banditpam_build_medoids_idx = kmed.medoids
            banditpam_build_medoids = data[banditpam_build_medoids_idx, :]

            banditpam_medoids_ref_cost_distance_matrix = distance_matrix(banditpam_build_medoids, data)
            banditpam_objective = np.sum(np.min(banditpam_medoids_ref_cost_distance_matrix, 0))

            losses += banditpam_objective,
            medoids += banditpam_build_medoids_idx,

    return np.array(losses), medoids


def find_loss_mistmatch(dataset_name,
                        num_data_list,
                        n_medoids,
                        algorithms=None,
                        loss: str = "L2",
                        verbose=True,
                        save_logs=True,
                        cache_width=1000,
                        dirname="scaling_with_n",
                        parallelize=False,
                        build_confidence=3,
                        n_swaps=0,
                        num_experiments=3):
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

    # exp 1, seed 5: original is worse than va and sklearn
    dataset = read_dataset(dataset_name)
    log_dir = os.path.join("logs", dirname)

    print("Running sampling complexity experiment with n on ", dataset_name)
    algorithms += "SKLEARN",

    for experiment_index in tqdm(range(num_experiments), desc="Experiment"):
        for num_data in tqdm(num_data_list, desc="Num data", leave=False):
            data_indices = np.random.randint(0, len(dataset), num_data)
            print(data_indices[0])
            data = dataset[data_indices]
            losses = {}
            medoids = {}

            for algorithm in algorithms:
                if algorithm == "SKLEARN":
                    kmedoids = KMedoids(n_clusters=n_medoids, method="pam", init="build", max_iter=n_swaps).fit(data)
                    sklearn_build_medoids = data[kmedoids.medoid_indices_, :]

                    sklearn_medoids_ref_cost_distance_matrix = distance_matrix(sklearn_build_medoids, data)
                    sklearn_objective = np.sum(np.min(sklearn_medoids_ref_cost_distance_matrix, 0))
                    losses[algorithm] = sklearn_objective
                    medoids[algorithm] = kmedoids.medoid_indices_
                else:
                    kmed, runtime = run_banditpam(algorithm, data, n_medoids, loss, cache_width, parallelize, n_swaps,
                                                  build_confidence)
                    banditpam_build_medoids_idx = kmed.build_medoids
                    banditpam_build_medoids = data[banditpam_build_medoids_idx, :]

                    banditpam_medoids_ref_cost_distance_matrix = distance_matrix(banditpam_build_medoids, data)
                    banditpam_objective = np.sum(np.min(banditpam_medoids_ref_cost_distance_matrix, 0))

                    losses[algorithm] = banditpam_objective
                    medoids[algorithm] = kmed.medoids

            # Report if the losses are not same
            if len(set(losses.values())) != 1:
                print("Experiment Index: ", experiment_index)
                print("Num data: ", num_data)

                for (i, algo) in enumerate(algorithms):
                    print(f"<{algo}>")
                    print("Loss   : ", losses[algo])
                    print("Medoid : ", medoids[algo])


def debug_loss_with_n(dataset_name,
                      num_data_list,
                      n_medoids,
                      algorithms=None,
                      loss: str = "L2",
                      verbose=True,
                      save_logs=True,
                      cache_width=1000,
                      dirname="scaling_with_n",
                      parallelize=False,
                      n_swaps=0,
                      num_experiments=3):
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
    algorithms += "SKLEARN",

    # exp 1, num 3000

    for experiment_index in range(num_experiments):
        print("Exp: ", experiment_index)

        for num_data in num_data_list:
            data_indices = np.random.randint(0, len(dataset), num_data)
            data = dataset[data_indices]
            losses = {}
            medoids = {}

            for algorithm in algorithms:
                print("Algorithm: ", algorithm)
                if algorithm == "SKLEARN":
                    kmedoids = KMedoids(n_clusters=n_medoids, method="pam", init="build", max_iter=n_swaps).fit(data)
                    sklearn_build_medoids = data[kmedoids.medoid_indices_, :]

                    sklearn_medoids_ref_cost_distance_matrix = distance_matrix(sklearn_build_medoids, data)
                    sklearn_objective = np.sum(np.min(sklearn_medoids_ref_cost_distance_matrix, 0))
                    losses[algorithm] = sklearn_objective
                    medoids[algorithm] = kmedoids.medoid_indices_
                else:
                    kmed, runtime = run_banditpam(algorithm, data, n_medoids, loss, cache_width, parallelize, n_swaps)
                    banditpam_build_medoids_idx = kmed.build_medoids
                    banditpam_build_medoids = data[banditpam_build_medoids_idx, :]

                    banditpam_medoids_ref_cost_distance_matrix = distance_matrix(banditpam_build_medoids, data)
                    banditpam_objective = np.sum(np.min(banditpam_medoids_ref_cost_distance_matrix, 0))

                    losses[algorithm] = banditpam_objective
                    medoids[algorithm] = kmed.medoids

            print("")

            # Report if the losses are not same
            if len(set(losses.values())) != 1:
                print("Experiment Index: ", experiment_index)
                print("Num data: ", num_data)

                for (i, algo) in enumerate(algorithms):
                    print(f"<{algo}>")
                    print("Loss   : ", losses[algo])
                    print("Medoid : ", medoids[algo])

                break
