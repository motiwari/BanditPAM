from banditpam import KMedoids
import numpy as np
import os

kmed_5 = KMedoids(
    n_medoids=5,
    algorithm="BanditPAM",
)
X1k = np.loadtxt(os.path.join("data", "MNIST_1k.csv"))
# print(X1k)
# arr = np.array([[1, 2, 3], [4, 5, 6]])
print("python beginning")
# kmed_5.build_medoids
# kmed_5.getDistanceComputations(True)
# kmed_5.misc_distance_computations
# kmed_5.build_distance_computations
# kmed_5.swap_distance_computations
# kmed_5.cache_writes
# kmed_5.cache_hits
# kmed_5.cache_misses
kmed_5.fit(X1k, "L2") # FAILS
# kmed_5.fit(arr, "L2") # FAILS
# kmed_5.labels
# kmed_5.average_loss
# kmed_5.build_loss
# kmed_5.medoids
# kmed_5.steps
# kmed_5.total_swap_time
# kmed_5.time_per_swap # FAILS because steps is 0, so divide by 0 error, this might be fixed if fit comes first
print("python end")

# inside beginning medoids_python
# inside end medoids_python
# inside beginning fit_python
# inside end fit_python
# [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# ...
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]]
# before
# inside beginning getMedoidsFinalPython
# inside else
# right before return ans
# after


# inside beginning medoids_python
# inside end medoids_python
# inside beginning fit_python
# inside end fit_python
# [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# ...
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]]
# before
# inside beginning fitPython
# fitting with no dist mat beginning
# end of kmedoids_algorithm.cpp fit method
# L2
