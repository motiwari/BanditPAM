# Use this script to verify BanditPAM is multithreaded
# Should complete in <= 3 seconds

import time
import numpy as np
import banditpam
import sys

X = np.loadtxt('data/MNIST-1k.csv')
kmed = banditpam.KMedoids(n_medoids = 5, algorithm = "BanditPAM")
start = time.time()
kmed.fit(X, "L2")
print(time.time() - start, "seconds")
print(kmed.medoids)
