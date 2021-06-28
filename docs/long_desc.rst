BanditPAM: A state-of-the-art, high-performance k-medoids algorithm
===================================================================

Introduction
------------
Clustering algorithms such as k-means are ubiquitous in modern data science applications. 
Despite its popularity, however, k-means has several drawbacks. Firstly, k-means only supports certain distance metrics, 
while different distance metrics may be desirable in other applications. For example, L1 norm and cosine distance are often used in
recommendation systems and single-cell RNA-seq analysis. Secondly, the cluster center of k-means is in general not a point 
in the dataset and hence lack of interpretability. This is problemetic when the problem requires interpretability on the center. 
For example, when the data is structured, such as images in computer vision where the mean image is virtually random noise.
Alternatively, k-medoids uses the points in the dataset itself -- the medoids -- as cluster centers. 
This enables interpretability of the cluster centers. Furthermore, k-medoids supports arbitrary dissimilarity functions 
in place of a distance metric.

Details
-------
Despite its advantages, k-medoids clustering is less popular than k-means due to its computational cost. 
The current state-of-the-art k-medoids algorithms PAM and FastPAM scale quadratically in the dataset size in each iterations
but they are still significantly slower than k-means, which scale linearly in dataset size in each iteration.
In this package, we provide a high-performance implementation of BanditPAM, 
a state-of-the-art k-medoids algorithm. The main contributions of BanditPAM are as follows:

* BanditPAM matches state-of-the-art in clustering quality but improves the runtime of previous approaches from O(n^2) to O(nlogn) in each iteration. 
* BanditPAM supports arbitrarily dissimilarity functions between points -- these functions need not even be metrics. 

This implementation is written in C++ for performance, but is callable from Python via Python bindings. The algorithm is empirically
validated on several large, real-word datasets with a reduction of distance evaluation up to 200x while returning the same results as 
state-of-the-art. For instance, BanditPAM scales linearly on MNIST dataset with L2 distance and returns the same solution as PAM.
On the scRNA-seq dataset with L1 distance, BanditPAM also scales almost linearly. These results confirm that BanditPAM takes almost 
linear number of distance evaluations per iteration for different datasets and different distance metrics. For further details, please see the paper at https://arxiv.org/abs/2006.06856 for the full paper and the code 
at https://github.com/ThrunGroup/BanditPAM/blob/main/README.md

If you use this software, please cite:
Mo Tiwari, Martin Jinye Zhang, James Mayclin, Sebastian Thrun, Chris Piech, Ilan Shomorony. "Bandit-PAM: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits" Advances in Neural Information Processing Systems (NeurIPS) 2020.