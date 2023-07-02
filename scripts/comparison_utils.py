from banditpam import KMedoids


def print_results(kmed: KMedoids, runtime: float):
    complexity_with_caching = kmed.getDistanceComputations(True) - kmed.cache_hits
    print("-----Results-----")
    print("Algorithm:", kmed.algorithm)
    print("Final Medoids:", kmed.medoids)
    print("Loss:", kmed.average_loss)
    print("Misc complexity:", f"{kmed.misc_distance_computations:,}")
    print("Build complexity:", f"{kmed.build_distance_computations:,}")
    print("Swap complexity:", f"{kmed.swap_distance_computations:,}")
    print("Number of Swaps", kmed.steps)
    print("Cache Writes: {:,}".format(kmed.cache_writes))
    print("Cache Hits: {:,}".format(kmed.cache_hits))
    print("Cache Misses: {:,}".format(kmed.cache_misses))
    print(
        "Total complexity (without misc):",
        f"{kmed.getDistanceComputations(False):,}",
    )
    print(
        "Total complexity (with misc):",
        f"{kmed.getDistanceComputations(True):,}",
    )
    print(
        "Total complexity (with caching):",
        f"{complexity_with_caching:,}",
    )
    print("Total runtime:", runtime)
