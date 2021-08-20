from BanditPAM import KMedoids
import numpy as np
import matplotlib.pyplot as plt
import BanditPAM
BanditPAM.set_num_threads(1)

def my_func():
    print('****************got called*************')


# Generate data from a Gaussian Mixture Model with the given means:
np.random.seed(0)
n_per_cluster = 40
means = np.array([[0,0], [-5,5], [5,5]])
X = np.vstack([np.random.randn(n_per_cluster, 2) + mu for mu in means])

# Fit the data with BanditPAM:
kmed = KMedoids(n_medoids = 3, algorithm = "BanditPAM")
# Writes results to gmm_log
#kmed.fit(X, 'L2', "gmm_log", modPath="multiply", dist_mat= "multiply", k=3)
kmed.fit(X, 'custom', "gmm_log", modPath="multiply", dist_mat= "multiply", k=3)

# Visualize the data and the medoids:
for p_idx, point in enumerate(X):
    if p_idx in map(int, kmed.medoids):
        plt.scatter(X[p_idx, 0], X[p_idx, 1], color='red', s = 40)
    else:
        plt.scatter(X[p_idx, 0], X[p_idx, 1], color='blue', s = 10)


plt.show()