import numpy as np
import os

from run_all_versions import run_banditpam
from scripts.comparison_utils import print_results, store_results
from scripts.constants import (
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_NO_CACHING,
    BANDITPAM_VA_CACHING,
    ALL_BANDITPAMS,
    MNIST,
    SCRNA,
)


def get_dataset_filename(dataset_name):
    if dataset_name == MNIST:
        return "MNIST_70k"
    elif dataset_name == SCRNA:
        return "scrna_reformat"
    else:
        assert False, "Invalid dataset name"


def run_sampling_complexity_experiment_with_k(dataset_name,
                                              n_medoids_list=None,
                                              algorithms=None,
                                              loss: str = "L2",
                                              verbose=True,
                                              save_logs=True,
                                              num_experiments=3):
    dataset_file_name = get_dataset_filename(dataset_name)
    dataset = np.loadtxt(os.path.join("data", f"{dataset_file_name}.csv"))
    num_data = len(dataset)
    log_dir = os.path.join("scripts", "experiments", "logs", "scaling_with_k_cluster")
    print("Running sampling complexity experiment with k")
    print("Dataset: ", dataset_name)

    for experiment_index in range(num_experiments):
        for n_medoids in n_medoids_list:
            for algorithm in algorithms:
                print("Running ", algorithm)

                log_name = f"{algorithm}_{dataset_name}_n{num_data}"
                kmed, runtime = run_banditpam(algorithm, dataset, n_medoids, loss)

                if verbose:
                    print_results(kmed, runtime)

                if save_logs:
                    store_results(kmed, runtime, log_dir, log_name, num_data, n_medoids)

                print("\n" * 2)

            print("\n" * 3)


def run_sampling_complexity_experiment_with_n(dataset_name,
                                              num_data_list,
                                              n_medoids,
                                              algorithms=None,
                                              loss: str = "L2",
                                              verbose=True,
                                              save_logs=True,
                                              num_experiments=3):
    dataset_file_name = get_dataset_filename(dataset_name)
    dataset = np.loadtxt(os.path.join("data", f"{dataset_file_name}.csv"))
    log_dir = os.path.join("scripts", "experiments", "logs", "scaling_with_n_cluster")

    print("Running sampling complexity experiment with n")
    print("Dataset: ", dataset_name)

    for experiment_index in range(num_experiments):
        for num_data in num_data_list:
            data_indices = np.random.randint(0, len(dataset), num_data)
            data = dataset[data_indices]
            for algorithm in algorithms:
                print("Running ", algorithm)
                log_name = f"{algorithm}_{dataset_name}_k{n_medoids}"
                kmed, runtime = run_banditpam(algorithm, data, n_medoids, loss)

                if verbose:
                    print_results(kmed, runtime)

                if save_logs:
                    store_results(kmed, runtime, log_dir, log_name, num_data, n_medoids)

                print("\n" * 2)

            print("\n" * 3)


def main():
    # run_sampling_complexity_experiment_with_n(SCRNA,
    #                                           num_data_list=[10],
    #                                           n_medoids=5, num_experiments=1)
    run_sampling_complexity_experiment_with_n(MNIST,
                                              num_data_list=np.linspace(10000, 70000, 5).astype(int),
                                              n_medoids=5, num_experiments=4)
    run_sampling_complexity_experiment_with_n(MNIST,
                                              num_data_list=np.linspace(10000, 70000, 5).astype(int),
                                              n_medoids=10, num_experiments=4)
    run_sampling_complexity_experiment_with_k(MNIST, n_medoids_list=[5, 10, 15, 20])


if __name__ == "__main__":
    main()
