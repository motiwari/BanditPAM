from sklearn.datasets import fetch_openml
from collections import defaultdict
import time
import numpy as np
import banditpam

from sklearn.metrics.pairwise import euclidean_distances
from kmedoids import fasterpam

X = np.loadtxt("../data/MNIST_1k.csv")

k = 20
kmed = banditpam.KMedoids(n_medoids=k, algorithm="FasterPAM")
start = time.time()
kmed.fit(X, "L2")
print(time.time() - start, "seconds")
print(kmed.medoids)
print("Loss:", kmed.average_loss)
print("Swap steps:", kmed.steps)
print("Done with FasterPAM")

kmed = banditpam.KMedoids(n_medoids=k, algorithm="BanditFasterPAM")
start = time.time()
kmed.fit(X, "L2")
print(time.time() - start, "seconds")
print(kmed.medoids)
print("Loss:", kmed.average_loss)
print("Swap steps:", kmed.steps)
print("Done with BanditFasterPAM")


# kmed = banditpam.KMedoids(n_medoids=k, algorithm="FastPAM1")
# start = time.time()
# kmed.fit(X, "L2")
# print(time.time() - start, "seconds")
# print(kmed.medoids)
# print(kmed.average_loss)

kmed = banditpam.KMedoids(n_medoids=k, algorithm="BanditPAM")
start = time.time()
kmed.fit(X, "L2")
print(time.time() - start, "seconds")
print(kmed.medoids)
print("Loss:", kmed.average_loss)
print("Swap steps:", kmed.steps)
print("Done with BanditPAM")