# Use this script to verify BanditPAM is multithreaded
# Should complete in <= 3 seconds

import time
import numpy as np
import banditpam

# banditpam.set_num_threads(1)

X = np.loadtxt('data/scrna_1k.csv', delimiter=',')

results = []

for i in range(10):    
    kmed = banditpam.KMedoids(n_medoids=10, algorithm="BanditPAM")
    start = time.time()
    kmed.fit(X, "L2")
    time_elapsed = time.time() - start
    results += time_elapsed,
    print((i+1), '/', 10, ' : ', time_elapsed, "seconds")

print('mean: ', np.mean(results))
# print(kmed.medoids)
