def print_results(kmed, runtime):
    print("-----Results-----")
    print("Algorithm:", kmed.algorithm)
    print("Final Medoids:", kmed.medoids)
    print("Loss:", kmed.average_loss)
    print("Misc complexity:", f"{kmed.misc_distance_computations:,}")
    print("Build complexity:", f"{kmed.build_distance_computations:,}")
    print("Swap complexity:", f"{kmed.swap_distance_computations:,}")
    print("Number of Swaps", kmed.steps)
    print(
        "Average Swap Sample Complexity:",
        f"{kmed.swap_distance_computations / kmed.steps:,}",
    )
    print(
        "Total complexity (without misc):",
        f"{kmed.getDistanceComputations(False):,}",
    )
    print(
        "Total complexity (with misc):",
        f"{kmed.getDistanceComputations(True):,}",
    )
    print("Runtime per swap:", runtime / kmed.steps)
    print("Total runtime:", runtime)
