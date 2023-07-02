import numpy as np
import os

import banditpam
import comparison_utils


def main():
    banditpam.set_num_threads(1)  # Turn off parallelization
    X = np.loadtxt(os.path.join("data", "MNIST_10k.csv"))

    for k in [5, 10, 20, 40, 80]:
        kmed = banditpam.KMedoids(
            n_medoids=k,
            algorithm="BanditPAM",
            use_cache=False,
            use_perm=False,
            parallelize=False,
        )
        kmed.fit(X, "L2")
        comparison_utils.print_results(kmed)


if __name__ == "__main__":
    main()
