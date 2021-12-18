BanditPAM: A state-of-the-art, high-performance k-medoids algorithm
===================================================================

Quickstart
----------

Run `pip3 install banditpam` and then check out the `examples <https://github.com/ThrunGroup/BanditPAM#example-1-synthetic-data-from-a-gaussian-mixture-model>`_.


Introduction
------------
Clustering algorithms are ubiquitous in modern data science applications. Compared to the commonly used k-means clustering, k-medoids clustering 
requires the cluster centers to be actual datapoints and supports arbitrary dissimilarity functions
in addition to standard distance metrics. This enables greater interpretability of the cluster centers and the clustering of structured 
objects. Despite these advantages, k-medoids clustering has been far less popular than k-means due to its computational cost.
We present BanditPAM, a randomized algorithm inspired by techniques from multi-armed bandits, 
that scales almost linearly with dataset size and runs significantly faster than prior algorithms while still matching the best prior algorithms in clustering quality. 
Prior k-medoids algorithms such as PAM and FastPAM scale quadratically in the dataset size and are significantly slower than k-means, which scales linearly in dataset size. 
This algorithm, BanditPAM, almost matches k-means in complexity and is significantly faster than prior state-of-the-art with runtime O(nlogn).

Details
-------
In this package, we provide a high-performance implementation of BanditPAM, a state-of-the-art k-medoids algorithm. 
BanditPAM:

* matches the best prior work in clustering quality but scales as O(nlogn) instead of O(n^2) in each iteration, and 
* supports arbitrary dissimilarity functions between points -- these functions need not even be metrics
  
This implementation is written in C++ for performance, but is callable from Python via Python bindings. The algorithm is empirically
validated on several large, real-word datasets with a reduction in the number of distance evaluations of up to 200x while returning the same results as 
prior state-of-the-art. For further details, please see the `full paper <https://arxiv.org/abs/2006.06856>`_ and the `code <https://github.com/ThrunGroup/BanditPAM/>`_.

If you use this software, please cite:
Mo Tiwari, Martin Jinye Zhang, James Mayclin, Sebastian Thrun, Chris Piech, Ilan Shomorony. "BanditPAM: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits" Advances in Neural Information Processing Systems (NeurIPS) 2020.