import banditpam
import numpy as np

def print_results(kmed):
    print("-----Results-----")
    print(kmed.algorithm)
    print(kmed.medoids)
    print("Loss:", kmed.average_loss)
    print("Misc complexity:", kmed.misc_distance_computations)
    print("Build complexity:", kmed.build_distance_computations)
    print("Swap complexity:", kmed.swap_distance_computations)
    print("Number of Swaps", kmed.steps)
    print("Total complexity:", kmed.getDistanceComputations(True))
    print("Total complexity:", kmed.getDistanceComputations(False))


def main():
    banditpam.set_num_threads(1)
    X = np.loadtxt("data/MNIST_1k.csv")

    kmed = banditpam.KMedoids(n_medoids=10, algorithm="BanditPAM", use_cache=False, use_perm=False, parallelize=False)
    kmed.fit(X, "L2")
    print_results(kmed)

    kmed = banditpam.KMedoids(n_medoids=10, algorithm="BanditPAM_v3", use_cache=False, use_perm=False, parallelize=False)
    kmed.fit(X, "L2")
    print_results(kmed)



if __name__ == "__main__":
    main()