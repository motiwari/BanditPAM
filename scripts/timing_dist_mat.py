# Use this script to verify BanditPAM is multithreaded
# Should complete in <= 3 seconds

import time
import numpy as np
import banditpam
from sklearn.metrics.pairwise import euclidean_distances


# banditpam.set_num_threads(1)

X = np.loadtxt("data/MNIST_10k.csv")
kmed = banditpam.KMedoids(n_medoids=5, algorithm="BanditPAM")
start = time.time()
diss = euclidean_distances(X)
kmed.fit(X, "L2", dist_mat=diss)
print(time.time() - start, "seconds")
print("Number of SWAP steps:", kmed.steps)
print(kmed.medoids)
