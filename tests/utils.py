MNIST_K_SCHEDULE = [4, 6, 8, 10]
NUM_SMALL_CASES = 10
NUM_LARGE_CASES = 48
SMALL_SAMPLE_SIZE = 100
PROPORTION_PASSING = 0.9

from banditpam import KMedoids

def on_the_fly(k, data, loss):
    kmed_bpam = KMedoids(n_medoids=k, algorithm="BanditPAM")
    kmed_naive = KMedoids(n_medoids=k, algorithm="naive")
    
    kmed_bpam.fit(data, loss)
    kmed_naive.fit(data, loss)
    return 1 if (sorted((kmed_bpam.medoids.tolist())) == sorted(kmed_naive.medoids.tolist()) and \
        sorted(kmed_bpam.build_medoids.tolist()) == sorted(kmed_naive.build_medoids.tolist())) else 0
