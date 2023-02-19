import numpy as np
import time
import os
from banditpam import KMedoids
import math

from tests.constants import MILLISECONDS_IN_A_SECOND


def time_measured_fit(
    kmed: KMedoids,
    X: np.array,
    loss: str = "L2",
):
    start = time.time()
    kmed.fit(X, loss)
    elapsed_time = time.time() - start  # Elapsed time in seconds
    return elapsed_time


def get_cache_statistics(
    kmed: KMedoids,
    X: np.array,
    loss: str = "L2",
    cache_width: int = 1000,
):
    kmed.cache_width = cache_width
    time = time_measured_fit(kmed, X, loss)
    print("Cache Width: {:,}".format(kmed.cache_width))
    print("Distance Computations: {:,}".format(kmed.distance_computations))
    print("Swap Steps: {:,}".format(kmed.steps))
    print("Cache Writes: {:,}".format(kmed.cache_writes))
    print("Cache Hits: {:,}".format(kmed.cache_hits))
    print("Cache Misses: {:,}".format(kmed.cache_misses))
    return (
        time,
        kmed.cache_width,
        kmed.steps,
        kmed.distance_computations,
        kmed.cache_writes,
        kmed.cache_hits,
        kmed.cache_misses,
    )


def main():
    def test_cache_stats():
        X = np.loadtxt(os.path.join("data", "MNIST_1k.csv"))
        kmed = KMedoids(n_medoids=5, algorithm="BanditPAM")

        (
            time_1000,
            width_1000,
            writes_1000,
            hits_1000,
            misses_1000,
        ) = get_cache_statistics(kmed=kmed, X=X, loss="L2", cache_width=1000)
        time_750, width_750, writes_750, hits_750, misses_750 = \
            get_cache_statistics(kmed=kmed, X=X, loss="L2", cache_width=750)
        time_500, width_500, writes_500, hits_500, misses_500 = \
            get_cache_statistics(kmed=kmed, X=X, loss="L2", cache_width=500)
        time_250, width_250, writes_250, hits_250, misses_250 = \
            get_cache_statistics(kmed=kmed, X=X, loss="L2", cache_width=250)
        time_0, width_0, writes_0, hits_0, misses_0 = \
            get_cache_statistics(kmed=kmed, X=X, loss="L2", cache_width=0)

        assert (
            hits_1000 > hits_750 > hits_500 > hits_250 > hits_0
        ), "Cache hits should increase as cache size increases"
        assert (
            misses_1000 < misses_750 < misses_500 < misses_250 < misses_0
        ), "Cache misses should decrease as cache size increases"
        assert (
            misses_1000 == 0
        ), "There should be no cache misses at width=1000 as the" \
           " whole dataset fits in memory"
        assert (
            hits_0 == 0
        ), "There should be no cache hits at width=0 as there is no cache"
        assert (
            writes_0 == 0
        ), "There should be no cache writes at width=0 as there is no cache"
        assert width_0 == 0, "Cache width should be 0 when set to 0"
        assert width_250 == 250, "Cache width should be 250 when set to 250"
        assert width_500 == 500, "Cache width should be 500 when set to 500"
        assert width_750 == 750, "Cache width should be 750 when set to 750"
        assert width_1000 == 1000, "Cache width should be 1000 when set " \
                                   "to 1000"

    def test_parallelization():
        X = np.loadtxt(os.path.join("data", "MNIST_10k.csv"))
        kmed = KMedoids(n_medoids=5, algorithm="BanditPAM")
        time_parallel = time_measured_fit(kmed=kmed, X=X, loss="L2")
        kmed.parallelize = False
        time_no_parallel = time_measured_fit(kmed=kmed, X=X, loss="L2")

        print(time_no_parallel, time_parallel)
        assert (
            time_no_parallel > time_parallel
        ), "Parallelization should increase speed on MNIST-10k dataset"

    def test_permutation():
        X = np.loadtxt(os.path.join("data", "MNIST_10k.csv"))
        kmed = KMedoids(n_medoids=5, algorithm="BanditPAM")
        time_perm = time_measured_fit(kmed=kmed, X=X, loss="L2")
        kmed.use_perm = False
        time_no_perm = time_measured_fit(kmed=kmed, X=X, loss="L2")

        print(time_perm, time_no_perm)
        assert (
            time_no_perm > time_perm
        ), "Permutation should increase speed on MNIST-10k dataset"

    def test_time_per_swap():
        X = np.loadtxt(os.path.join("data", "MNIST_10k.csv"))
        kmed = KMedoids(n_medoids=5, algorithm="BanditPAM")
        time_ = time_measured_fit(kmed=kmed, X=X, loss="L2")
        total_swap_time = kmed.total_swap_time
        average_swap_time = kmed.time_per_swap

        print(
            time_,
            total_swap_time / MILLISECONDS_IN_A_SECOND,
            average_swap_time / MILLISECONDS_IN_A_SECOND,
        )
        assert (
            time_ > total_swap_time / MILLISECONDS_IN_A_SECOND
        ), "Total time should be greater than total swap time"
        assert (
            total_swap_time > average_swap_time / MILLISECONDS_IN_A_SECOND
        ), "Total swap time should be greater than average swap time"
        assert math.isclose(
            total_swap_time / kmed.steps, average_swap_time, abs_tol=kmed.steps
        ), "Average swap time inconsistent with total swap time and steps"

    def test_old_bpam_vs_new_bpam():
        X = np.loadtxt(os.path.join("data", "MNIST_1k.csv"))
        kmed = KMedoids(
            n_medoids=10,
            algorithm="BanditPAM",
            use_cache=False,
            use_perm=False,
            parallelize=False,
        )
        (
            bpam_time,
            bpam_width,
            bpam_distance_computations,
            bpam_steps,
            bpam_writes,
            bpam_hits,
            bpam_misses,
        ) = get_cache_statistics(kmed=kmed, X=X, loss="L2", cache_width=1000)

        kmed_orig = KMedoids(
            n_medoids=10,
            algorithm="BanditPAM_orig",
            use_cache=False,
            use_perm=False,
            parallelize=False,
        )
        (
            bpam_orig_time,
            bpam_orig_width,
            bpam_orig_distance_computations,
            bpam_orig_steps,
            bpam_orig_writes,
            bpam_orig_hits,
            bpam_orig_misses,
        ) = get_cache_statistics(
            kmed=kmed_orig,
            X=X,
            loss="L2",
            cache_width=1000,
        )

        print(bpam_time, bpam_orig_time)
        print(kmed.time_per_swap, kmed_orig.time_per_swap)

    # test_cache_stats()
    # test_parallelization()
    # test_permutation()
    # test_time_per_swap()
    test_old_bpam_vs_new_bpam()


if __name__ == "__main__":
    main()
