import ast, sys, getopt
import time
import numpy as np
import banditpam
from typing import List, Tuple
from utils.experiment_utils import get_dataset, print_summary


def run_experiments(
    dataset: np.ndarray,
    n_experiments: int = 3,
    n_medoids: int = 5,
    useCacheP: bool = True,
    usePerm: bool = True,
    cache_multiplier: int = 4000,
) -> Tuple[float, float]:
    """
    Run one type of experiments on a given condition

    :param dataset: Numpy array dataset
    :param n_experiments: Number of experiments torun
    :param n_medoids: Number of medoids to find
    :param useCacheP: Whether to use cache
    :param usePerm: Whether to use permutation
    :param cache_multiplier: A multipler value that determines the cache size

    :return: Tuple of mean and standard deviation of run time
    """

    print("Cache: %r   Perm: %r" % (useCacheP, usePerm))
    results = []
    for seed in range(n_experiments):
        kmed = banditpam.KMedoids(
            n_medoids=n_medoids,
            algorithm="BanditPAM",
            useCacheP=useCacheP,
            usePerm=usePerm,
            cacheMultiplier=cache_multiplier,
            seed=seed,
        )
        start = time.time()
        kmed.fit(dataset, "L2")
        time_elapsed = time.time() - start
        results += (time_elapsed,)
        print((seed + 1), "/", n_experiments, " : ", time_elapsed, "seconds")

    mean = np.mean(results)
    std = np.std(results)

    return mean, std


def compare_cache_perm(
    dataset_name: str,
    n_experiments: int,
    n_data: int,
    n_medoids: int,
    cache_multiplier: int,
) -> List[int]:
    """
    Run three types of experiments on a given condition
    1. Without cache and permutation
    2. With cache and without permutation
    3. With cache and permutation

    :param dataset_name: Name of a dataset (currently only supports MNIST)
    :param n_experiments: Number of experiments to run
    :param n_data: Size of the dataset
    :param n_medoids: Number of medoids to find
    :param cache_multiplier: A multipler value that determines the cache size

    :return: List of three tuple results (mean, std) from each experiement
    """

    print(f"\n[{dataset_name}={n_data}, K={n_medoids}]")

    dataset = get_dataset(dataset_name=dataset_name, n_data=n_data)

    stats1 = run_experiments(
        dataset,
        n_experiments=n_experiments,
        n_medoids=n_medoids,
        useCacheP=False,
        usePerm=False,
        cache_multiplier=cache_multiplier,
    )

    stats2 = run_experiments(
        dataset,
        n_experiments=n_experiments,
        n_medoids=n_medoids,
        useCacheP=True,
        usePerm=False,
        cache_multiplier=cache_multiplier,
    )

    stats3 = run_experiments(
        dataset,
        n_experiments=n_experiments,
        n_medoids=n_medoids,
        useCacheP=True,
        usePerm=True,
        cache_multiplier=cache_multiplier,
    )

    stats = [stats1, stats2, stats3]
    return stats


def run_multiple_experiments(
    dataset_list: List[str],
    n_data_list: List[str],
    n_medoids_list=[],
    cache_multiplier=5000,
) -> None:
    """
    Run experiments on multiple conditions (datasets, number of data, number of medoids)

    :param dataset_list: List of names of datasets (currently only supports MNIST)
    :param n_data_list: List of numbers of data
    :param n_medoids_list: List of numbers of medoids to find
    :param cache_multiplier: A multipler value that determines the cache size
    """
    stats_list = []

    # collect experiment results
    print("\n" + "-" * 40)
    print("\nRUNNING EXPERIMENTS...")

    for dataset_name in dataset_list:
        for n_data in n_data_list:
            for n_medoids in n_medoids_list:
                stats = compare_cache_perm(
                    dataset_name=dataset_name,
                    n_experiments=3,
                    n_data=n_data,
                    n_medoids=n_medoids,
                    cache_multiplier=cache_multiplier,
                )
                stats_list += (stats,)

    # print results
    print("\n" + "-" * 40)
    print("\nPRINTING SUMMARY\n")

    i = 0
    for dataset_name in dataset_list:
        print(
            "{:30}{:30}{:30}".format(
                "Cache (X) Perm (X)",
                "Cache (O) Perm (X)",
                "Cache (O) Perm (O)",
            )
        )
        for n_data in n_data_list:
            for n_medoids in n_medoids_list:
                stats = stats_list[i]
                i += 1
                print_summary(stats, dataset_name, n_data, n_medoids)

    print("\n" + "-" * 40)


def main(argv):
    try:
        opts, _ = getopt.getopt(
            argv,
            "k:n:d:c:",
            ["n_medoids=", "n_data=", "dataset=", "cache_multiplier="],
        )

        dataset_list = ["mnist"]
        n_medoids_list = [5, 10]
        n_data_list = [10000, 30000]
        cache_multiplier = 5000

        for opt, arg in opts:
            if opt in ["-k", "--n_medoids"]:
                arg = ast.literal_eval(arg)
                n_medoids_list = [arg] if type(arg) == int else arg
            elif opt in ["-n", "--n_data"]:
                arg = ast.literal_eval(arg)
                n_data_list = [arg] if type(arg) == int else arg
            elif opt in ["-c", "--cache_multiplier"]:
                cache_multiplier = int(arg)
            # elif opt in ["-d", "--datasets"]:
            #     print(type(arg))
            #     arg = ast.literal_eval(arg)
            #     dataset_list = [arg] if type(arg) == str else arg
            else:
                assert (False, "Unhandled option")

        run_multiple_experiments(
            dataset_list, n_data_list, n_medoids_list, cache_multiplier
        )

    except getopt.GetoptError as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
