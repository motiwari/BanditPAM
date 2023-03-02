from sklearn.datasets import fetch_openml
from collections import defaultdict
import time
import numpy as np
import banditpam

from sklearn.metrics.pairwise import euclidean_distances
from kmedoids import fasterpam

X = np.loadtxt("../data/MNIST_1k.csv")

kmed = banditpam.KMedoids(n_medoids=5, algorithm="FasterPAM")
kmed.fit(X, "L2")
print(kmed.medoids)
print(kmed.average_loss)