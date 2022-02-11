# Use this script to verify BanditPAM is multithreaded
# Should complete in <= 3 seconds

import time
import numpy as np
import banditpam

# banditpam.set_num_threads(1)

X = np.loadtxt('data/MNIST_10k.csv')
kmed = banditpam.KMedoids(n_medoids=5, algorithm="BanditPAM")
start = time.time()
kmed.fit(X, "L2", np.array([[]]))
print(time.time() - start, "seconds")
print("Number of SWAP steps:", kmed.steps)
print(kmed.medoids)
