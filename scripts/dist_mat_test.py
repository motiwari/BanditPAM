import numpy as np
import banditpam

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

dist_mat1 = np.array(
    [
        [0, np.sqrt(27), np.sqrt(108)],
        [np.sqrt(27), 0, np.sqrt(27)],
        [np.sqrt(108), np.sqrt(27), 0],
    ]
)
kmed = banditpam.KMedoids(n_medoids=2, algorithm="BanditPAM")
kmed.fit(X, "L2", dist_mat=dist_mat1)
print("Number of SWAP steps:", kmed.steps)
print(kmed.medoids)
