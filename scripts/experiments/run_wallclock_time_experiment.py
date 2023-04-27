import numpy as np
import os
import banditpam
import time

import comparison_utils


def compare_wallclock_time(dataset_names: list, loss: str):
    datasets = [np.loadtxt(os.path.join("data", f"{dataset_name}.csv")) for dataset_name in dataset_names]

    for (dataset_name, X) in zip(dataset_names, datasets):
        print(f"======Dataset: {dataset_name}======")
        # BanditPAM VA + Caching
        print("Running BanditPAM VA + Caching")
        new_kmed = banditpam.KMedoids(
            n_medoids=10,
            algorithm="BanditPAM",
            use_cache=True,
            use_perm=True,
            parallelize=True,
        )
        start = time.time()
        new_kmed.fit(X, loss)
        end = time.time()
        comparison_utils.print_results(new_kmed, end - start)

        # BanditPAM VA
        print("\n\nBanditPAM VA")
        new_kmed = banditpam.KMedoids(
            n_medoids=10,
            algorithm="BanditPAM",
            use_cache=False,
            use_perm=False,
            parallelize=True,
        )
        start = time.time()
        new_kmed.fit(X, loss)
        end = time.time()
        comparison_utils.print_results(new_kmed, end - start)

        # BanditPAM Original Caching
        print("\n\nBanditPAM Original Caching")
        new_kmed = banditpam.KMedoids(
            n_medoids=10,
            algorithm="BanditPAM_orig",
            use_cache=True,
            use_perm=False,
            parallelize=True,
        )
        start = time.time()
        new_kmed.fit(X, loss)
        end = time.time()
        comparison_utils.print_results(new_kmed, end - start)

        # BanditPAM Original
        print("\n\nBanditPAM Original")
        old_kmed = banditpam.KMedoids(
            n_medoids=10,
            algorithm="BanditPAM_orig",
            use_cache=False,
            use_perm=False,
            parallelize=True,
        )
        start = time.time()
        old_kmed.fit(X, loss)
        end = time.time()
        comparison_utils.print_results(old_kmed, end - start)

        print("\n"*5)


def main():
    compare_wallclock_time(["MNIST_1k", "MNIST_70k"], "L2")
    # compare_wallclock_time(["scrna_reformat"], "L1")


if __name__ == "__main__":
    main()
