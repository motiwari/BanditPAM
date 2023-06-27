import time
import numpy as np
from sklearn.cluster import KMeans

X = np.loadtxt("data/MNIST_1k.csv")
kmeans = KMeans(
    n_clusters=5, init="random", n_init=1, copy_x=True, algorithm="full",
)
start = time.time()
kmeans.fit(X)
print(time.time() - start, "seconds")
print(kmeans.cluster_centers_)
