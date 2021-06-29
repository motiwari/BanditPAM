BanditPAM: A state-of-the-art, high-performance k-medoids algorithm
===================================================================

Introduction
------------
Clustering algorithms are ubiquitous in modern data science applications. Compared to k-means clustering, k-medoids clustering 
uses the points in the dataset itself -- the medoids -- as cluster centers and supports arbitrary dissimilarity functions
in place of a distance metric. This enables greater interpretability of the cluster centers and the clustering of structured 
objects. Despite its advantages, k-medoids clustering is less popular than k-means due to its computational cost. 
The k-medoids algorithms such as PAM and FastPAM scale quadratically in the dataset size in each iterations
but they are still significantly slower than k-means, which scales linearly in dataset size in each iteration. In this work,
we introduce BanditPAM, a randomized algorithm inspired by techniques from multi-armed bandits, that runs significantly faster
than prior k-medoids algorithms and achieves the same clustering results. 

Details
-------
In this package, we provide a high-performance implementation of BanditPAM, a state-of-the-art k-medoids algorithm. 
The main contributions of BanditPAM are as follows:

* BanditPAM matches the best prior work in clustering quality but scales as O(nlogn) instead of O(n^2) in each iteration.
* BanditPAM supports arbitrary dissimilarity functions between points -- these functions need not even be metrics. 

This implementation is written in C++ for performance, but is callable from Python via Python bindings. The algorithm is empirically
validated on several large, real-word datasets with a reduction of distance evaluation up to 200x while returning the same results as 
prior work. These results confirm that BanditPAM takes almost linear number of distance evaluations per iteration for different datasets 
and different distance metrics. For further details, please see the paper at https://arxiv.org/abs/2006.06856 for the full paper and the code 
at https://github.com/ThrunGroup/BanditPAM/

If you use this software, please cite:
Mo Tiwari, Martin Jinye Zhang, James Mayclin, Sebastian Thrun, Chris Piech, Ilan Shomorony. "Bandit-PAM: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits" Advances in Neural Information Processing Systems (NeurIPS) 2020.