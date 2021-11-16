# Use this script as a hacky way to verify BanditPAM is multithreaded
# Should complete in <= 3 seconds

import time
import numpy as np
import BanditPAM
import sys

#print(BanditPAM.sum_thread_ids())
#BanditPAM.set_num_threads(int(sys.argv[1]))
#print(BanditPAM.sum_thread_ids())

X = np.loadtxt('data/MNIST-10k.csv')
kmed = BanditPAM.KMedoids(n_medoids = 5, algorithm = "BanditPAM")
start = time.time()
kmed.fit(X, "L2", "gmm_log")
print(time.time() - start, "seconds")
print(kmed.medoids)
