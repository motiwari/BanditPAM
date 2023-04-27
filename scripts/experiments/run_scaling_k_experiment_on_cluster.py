import numpy as np
import os
import banditpam
import time

from run_all_versions import run_banditpam
from scripts.comparison_utils import print_results, store_results
from scripts.constants import (
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_NO_CACHING,
    BANDITPAM_VA_CACHING,
    ALL_BANDITPAMS,
    MNIST,
)

from run_scaling_experiment import run_sampling_complexity_experiment_with_k


def main():
    run_sampling_complexity_experiment_with_k(MNIST, algorithms=[BANDITPAM_VA_CACHING], n_medoids_list=[20])


if __name__ == "__main__":
    main()
