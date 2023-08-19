import numpy as np
import os
import banditpam
import time

import comparison_utils


def main():
    X70k = np.loadtxt(os.path.join("data", "MNIST_70k.csv"))

    new_kmed = banditpam.KMedoids(
        n_medoids=5,
        algorithm="BanditFasterPAM",
        max_iter=70000,
    )
    start = time.time()
    new_kmed.fit(X70k, "L2")
    end = time.time()
    comparison_utils.print_results(new_kmed, end - start)

    old_kmed = banditpam.KMedoids(
        n_medoids=5,
        algorithm="BanditPAM",
        max_iter=1,
    )
    start = time.time()
    old_kmed.fit(X70k, "L2")
    end = time.time()
    comparison_utils.print_results(old_kmed, end - start)


if __name__ == "__main__":
    main()
