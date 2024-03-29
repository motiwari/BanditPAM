import numpy as np
import os
import banditpam
import time

import comparison_utils


def main():
    X1k = np.loadtxt(os.path.join("data", "MNIST_1k.csv"))
    X10k = np.loadtxt(os.path.join("data", "MNIST_10k.csv"))
    # X70k = np.loadtxt(os.path.join("data", "MNIST_70k.csv"))

    for X in [X1k, X10k]:  # , X10k, X70k]:
        new_kmed = banditpam.KMedoids(
            n_medoids=10,
            algorithm="BanditPAM",
            use_cache=False,
            use_perm=False,
            parallelize=False,
        )
        start = time.time()
        new_kmed.fit(X, "L2")
        end = time.time()
        comparison_utils.print_results(new_kmed, end - start)

        old_kmed = banditpam.KMedoids(
            n_medoids=10,
            algorithm="BanditPAM_orig",
            use_cache=False,
            use_perm=False,
            parallelize=False,
        )
        start = time.time()
        old_kmed.fit(X, "L2")
        end = time.time()
        comparison_utils.print_results(old_kmed, end - start)


if __name__ == "__main__":
    main()
