# banditpam <img src="man/figures/logo.png" align="right" />

We provide an R interface to the high-performance implementation of
[banditpam](https://proceedings.neurips.cc/paper/2020/file/73b817090081cef1bca77232f4532c5d-Paper.pdf),
a $k$-medoids clustering algorithm.

If you use this software, please cite:

>>Mo Tiwari, Martin Jinye Zhang, James Mayclin, Sebastian Thrun, Chris Piech, Ilan Shomorony. "banditpam: Almost Linear Time *k*-medoids Clustering via Multi-Armed Bandits" Advances in Neural Information Processing Systems (NeurIPS) 2020.

Here's a BibTeX entry:
```
@inproceedings{banditpam,
  title={banditpam: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits},
  author={Tiwari, Mo and Zhang, Martin J and Mayclin, James and Thrun, Sebastian and Piech, Chris and Shomorony, Ilan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={368--374},
  year={2020}
}
```

## Installation

banditpam can be installed from CRAN like any other
package. Development versions may be installed via:

``` r
remotes::install_github("bnaras/banditpam", subdir = "R_package/banditpam")
```

For the latter, you need the package development toolchain for R
packages. Refer to [CRAN](https://cran.r-project.org).


## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(banditpam)
## Generate data from a Gaussian Mixture Model with the given means:
set.seed(10)
n_per_cluster <- 40
means <- list(c(0, 0), c(-5, 5), c(5, 5))
X <- do.call(rbind, lapply(means, MASS::mvrnorm, n = n_per_cluster, Sigma = diag(2)))
## Create KMediods object
obj <- KMedoids$new(k = 3)
## Fit data
obj$fit(data = X, loss = "l2")
## Retrieve medoid indices
meds <- obj$get_medoids_final()
## Plot the results
plot(X[, 1], X[, 2])
points(X[meds, 1], X[meds, 2], col = "red", pch = 19)
##
## One can query some statistics too; see help("KMedoids")
##
obj$get_statistic("dist_computations")
obj$get_statistic("dist_computations_and_misc")
obj$get_statistic("cache_misses")

```

