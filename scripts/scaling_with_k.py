# Use this script to verify BanditPAM is multithreaded
# Should complete in <= 3 seconds
import time
import numpy as np
import banditpam

banditpam.set_num_threads(1)

X = np.loadtxt("data/MNIST_1k.csv")

for k in [5, 10, 30]:
    kmed = banditpam.KMedoids(
        n_medoids=k,
        algorithm="BanditPAM",
        use_cache=False,
        use_perm=False,
        parallelize=False,
    )
    start = time.time()
    kmed.fit(X, "L2")
    print(time.time() - start, "seconds")
    print("Number of SWAP steps:", kmed.steps)
    print(kmed.medoids)
