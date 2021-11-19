BanditPAM: A state-of-the-art, high-performance k-medoids algorithm
===================================================================

Quickstart
----------

Please see https://github.com/ThrunGroup/BanditPAM#python-quickstart.

Introduction
------------
Clustering algorithms are ubiquitous in modern data science applications. Compared to the commonly used k-means clustering, k-medoids clustering 
uses the points in the dataset itself -- the medoids -- as cluster centers and supports arbitrary dissimilarity functions
in addition to standard distance metrics. This enables greater interpretability of the cluster centers and the clustering of structured 
objects. We present BanditPAM, a randomized algorithm inspired by techniques from multi-armed bandits, that scales almost linearly 
with dataset size and runs significantly faster than prior algorithms while still matching the best prior algorithms in clustering quality. 
Despite its advantages, k-medoids clustering has been less popular than k-means due to its computational cost. 
Prior k-medoids algorithms such as PAM and FastPAM scale quadratically in the dataset size in each iterations
but they are still significantly slower than k-means, which scales linearly in dataset size in each iteration. This algorithm, 
BanditPAM, matches k-means in complexity and is significantly faster than prior state-of-the-art.

Details
-------
In this package, we provide a high-performance implementation of BanditPAM, a state-of-the-art k-medoids algorithm. 
The main contributions of BanditPAM are as follows:

* BanditPAM matches the best prior work in clustering quality but scales as O(nlogn) instead of O(n^2) in each iteration.
* BanditPAM supports arbitrary dissimilarity functions between points -- these functions need not even be metrics. 
  
This implementation is written in C++ for performance, but is callable from Python via Python bindings. The algorithm is empirically
validated on several large, real-word datasets with a reduction in the number of distance evaluations of up to 200x while returning the same results as 
prior state-of-the-art. For further details, please see the full paper at https://arxiv.org/abs/2006.06856 and the code at https://github.com/ThrunGroup/BanditPAM/

If you use this software, please cite:
Mo Tiwari, Martin Jinye Zhang, James Mayclin, Sebastian Thrun, Chris Piech, Ilan Shomorony. "Bandit-PAM: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits" Advances in Neural Information Processing Systems (NeurIPS) 2020.