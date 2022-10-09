import time
import numpy as np
import banditpam

def run_experiment(dataset, num_experiments=3, useCacheP=True, usePerm=True, cacheMultiplier=5000):
    print("---Cache: %r   Perm: %r---" % (useCacheP, usePerm))
    results = []
    for i in range(num_experiments):    
        kmed = banditpam.KMedoids(n_medoids=10, algorithm="BanditPAM", useCacheP=useCacheP, usePerm=usePerm, cacheMultiplier=cacheMultiplier)
        start = time.time()
        kmed.fit(dataset, "L2")
        time_elapsed = time.time() - start
        results += time_elapsed,
        print((i+1), '/', num_experiments, ' : ', time_elapsed, "seconds")

    print(">>> mean: ", np.mean(results), 'std: ', np.std(results), " <<<\n")

def run_experiments(dataset, num_experiments):
    run_experiment(dataset, num_experiments=num_experiments, useCacheP=True, usePerm=True)
    run_experiment(dataset, num_experiments=num_experiments, useCacheP=True, usePerm=False)
    run_experiment(dataset, num_experiments=num_experiments, useCacheP=False, usePerm=False)

print("\n\n**********DATASET: MNIST 3K**********")
dataset = np.loadtxt('data/MNIST_70k.csv', skiprows=70000-3000)
run_experiments(dataset, num_experiments=1)

print("\n\n**********DATASET: MNIST 6K**********")
dataset = np.loadtxt('data/MNIST_70k.csv', skiprows=70000-6000)
run_experiments(dataset, num_experiments=1)

print("\n\n**********DATASET: MNIST 10K**********")
dataset = np.loadtxt('data/MNIST_70k.csv', skiprows=70000-10000)
run_experiments(dataset, num_experiments=1)

print("\n\n**********DATASET: MNIST FULL**********")
dataset = np.loadtxt('data/MNIST_70k.csv')
run_experiments(dataset, num_experiments=1)

# print("**********DATASET: SCRNA**********")
# dataset = np.loadtxt('data/MNIST_FULL.csv', delimiter=',')
# run_experiment(dataset, num_experiments=3, useCacheP=True, usePerm=True)
# run_experiment(dataset, num_experiments=1, useCacheP=False, usePerm=False)