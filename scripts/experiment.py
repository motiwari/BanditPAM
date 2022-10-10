import ast, sys, getopt
import time
import numpy as np
import banditpam
from utils.experiment_utils import get_dataset, print_summary

def _run_experiments(dataset, n_experiments=3, n_medoids=5, useCacheP=True, usePerm=True, cacheMultiplier=5000):
    print("Cache: %r   Perm: %r" % (useCacheP, usePerm))
    results = []
    for i in range(n_experiments):    
        kmed = banditpam.KMedoids(n_medoids=n_medoids, algorithm="BanditPAM", useCacheP=useCacheP, usePerm=usePerm, cacheMultiplier=cacheMultiplier)
        start = time.time()
        kmed.fit(dataset, "L2")
        time_elapsed = time.time() - start
        results += time_elapsed,
        print((i+1), '/', n_experiments, ' : ', time_elapsed, "seconds")

    mean = np.mean(results)
    std = np.std(results)

    return mean, std

def run_experiments(dataset_name, n_experiments, n_data, n_medoids):
    print(f"\n[{dataset_name}={n_data}, K={n_medoids}]")
    dataset = get_dataset(dataset_name=dataset_name, n_data=n_data)
    stats1 = _run_experiments(dataset, n_experiments=n_experiments, n_medoids=n_medoids, useCacheP=False, usePerm=False)
    stats2 = _run_experiments(dataset, n_experiments=n_experiments, n_medoids=n_medoids, useCacheP=True, usePerm=False)
    stats3 = _run_experiments(dataset, n_experiments=n_experiments, n_medoids=n_medoids, useCacheP=True, usePerm=True)
    stats = [stats1, stats2, stats3]
    return stats

def run_multiple_experiments(dataset_list=[], n_data_list=[], n_medoids_list=[]):
    stats_list = []

    # collect experiment results
    print("\n" + "-"*40)
    print("\nRUNNING EXPERIMENTS...")

    for dataset_name in dataset_list:
        for n_data in n_data_list:
            for n_medoids in n_medoids_list:
                stats = run_experiments(dataset_name=dataset_name, n_experiments=3, n_data=n_data, n_medoids=n_medoids)
                stats_list += stats,

    # print results
    print("\n" + "-"*40)
    print("\nPRINTING SUMMARY\n")

    i = 0
    for dataset_name in dataset_list:
        print("{:30}{:30}{:30}".format('Cache (X) Perm (X)', 'Cache (O) Perm (X)', 'Cache (O) Perm (O)'))
        for n_data in n_data_list:
            for n_medoids in n_medoids_list:
                stats = stats_list[i]
                i += 1
                print_summary(stats, dataset_name, n_data, n_medoids)
    
    print("\n" + "-"*40)
    
def main(argv):
    try:
        opts, _ = getopt.getopt(argv, "k:n:d:", ["n_medoids=", "n_data=", "dataset="])
        
        dataset_list = ["mnist"]
        n_medoids_list = [5, 10]
        n_data_list = [10000, 30000]

        for opt, arg in opts:
            if opt in ["-k", "--n_medoids"]:
                arg = ast.literal_eval(arg)
                n_medoids_list = [arg] if type(arg) == int else arg
            elif opt in ["-n", "--n_data"]:
                arg = ast.literal_eval(arg)
                n_data_list = [arg] if type(arg) == int else arg
            # elif opt in ["-d", "--datasets"]:
            #     print(type(arg))
            #     arg = ast.literal_eval(arg)
            #     dataset_list = [arg] if type(arg) == str else arg
            else:
                assert(False, "Unhandled option")

        run_multiple_experiments(dataset_list, n_data_list, n_medoids_list)

    except getopt.GetoptError as error:
        print(error)
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])