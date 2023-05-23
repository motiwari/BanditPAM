import numpy as np

from scaling_experiment import scaling_experiment_with_k, scaling_experiment_with_n
from scripts.constants import (
    # Algorithms
    ALL_BANDITPAMS,

    # Datasets
    MNIST,
    CIFAR,
)


def get_loss_function(dataset):
    if dataset == MNIST:
        return "L1"
    else:
        return "L2"


def get_num_data_list(dataset):
    if dataset == MNIST:
        num_data = 70000
        return np.linspace(40000, num_data, 5)
    elif dataset == CIFAR:
        num_data = 40000
        return np.linspace(10000, num_data, 5)


def run_scaling_experiment_with_k():
    for dataset in [MNIST, CIFAR]:
        loss = get_loss_function(dataset)
        scaling_experiment_with_k(dataset_name=dataset,
                                  loss=loss,
                                  algorithms=ALL_BANDITPAMS,
                                  n_medoids_list=[5, 8, 10]
                                  )


def run_scaling_experiment_with_n():
    for dataset in [MNIST, CIFAR]:
        loss = get_loss_function(dataset)
        num_data_list = get_num_data_list(dataset)
        for n_medoids in [5, 10]:
            scaling_experiment_with_n(dataset_name=dataset,
                                      loss=loss,
                                      algorithms=ALL_BANDITPAMS,
                                      n_medoids=n_medoids,
                                      num_data_list=num_data_list,
                                      )


if __name__ == "__main__":
    run_scaling_experiment_with_k()
    run_scaling_experiment_with_n()
