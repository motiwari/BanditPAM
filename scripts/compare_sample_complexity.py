import banditpam
import numpy as np

def main():
    banditpam.set_num_threads(1)
    X = np.loadtxt("data/MNIST_1k.csv")
    kmed = banditpam.KMedoids(n_medoids=5, algorithm="BanditPAM", use_cache=False, use_perm=False, parallelize=False)
    kmed.fit(X, "L2")
    print("Misc complexity:", kmed.misc_distance_computations)
    print("Build complexity:", kmed.build_distance_computations)
    print("Swap complexity:", kmed.swap_distance_computations)
    print("Total complexity:", kmed.getDistanceComputations(True))
    print("Total complexity:", kmed.getDistanceComputations(False))



if __name__ == "__main__":
    main()