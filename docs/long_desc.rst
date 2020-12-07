BanditPAM: A state-of-the-art, high-performance k-medoids algorithm
===================================================================

Introduction
------------
Clustering algorithms such as k-means are ubiquitous in modern data science applications. Despite its popularity, however, k-means has several drawbacks. K-means only supports certain distance metrics, and its cluster centers can lack interpretability. Alternatively, k-medoids uses the points in the dataset itself -- the medoids -- as cluster centers. This enables interpretability of the cluster centers. Furthermore, k-medoids supports arbitrary dissimilarity functions in place of a distance metric.

Details
-------
In this package, we provide a high-performance implementation of BanditPAM , a state-of-the-art k-medoids algorithm. BanditPAM matches state-of-the-art in clustering quality but improves the runtime of previous approaches from O(n^2) to O(nlogn) in each iteration. Furthermore, BanditPAM supports arbitrarily dissimilarity functions between points -- these functions need not even be metrics. This implementation is written in C++ for performance, but is callable from Python via Python bindings. For further details, please see the paper at https://arxiv.org/abs/2006.06856 for the full paper and the code at https://github.com/ThrunGroup/BanditPAM/blob/main/README.md

If you use this software, please cite:
Mo Tiwari, Martin Jinye Zhang, James Mayclin, Sebastian Thrun, Chris Piech, Ilan Shomorony. "Bandit-PAM: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits" Advances in Neural Information Processing Systems (NeurIPS) 2020.