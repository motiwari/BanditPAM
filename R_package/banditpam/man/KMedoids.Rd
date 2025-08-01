% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/KMedoid.R
\name{KMedoids}
\alias{KMedoids}
\title{KMedoids Class}
\description{
This class wraps around the C++ KMedoids class and exposes methods and fields of the C++ object.
}
\examples{
# Generate data from a Gaussian Mixture Model with the given means:
set.seed(10)
n_per_cluster <- 40
means <- list(c(0, 0), c(-5, 5), c(5, 5))
X <- do.call(rbind, lapply(means, MASS::mvrnorm, n = n_per_cluster, Sigma = diag(2)))
obj <- KMedoids$new(k = 3)
obj$fit(data = X, loss = "l2")
meds <- obj$get_medoids_final()
plot(X[, 1], X[, 2])
points(X[meds, 1], X[meds, 2], col = "red", pch = 19)
}
\section{Active bindings}{
\if{html}{\out{<div class="r6-active-bindings">}}
\describe{
\item{\code{k}}{(\code{integer(1)})\cr
The number of medoids/clusters to create}

\item{\code{max_iter}}{(\code{integer(1)})\cr
max_iter the maximum number of SWAP steps the algorithm runs}

\item{\code{build_conf}}{(\code{integer(1)})\cr
Parameter that affects the width of BUILD confidence intervals, default 1000}

\item{\code{swap_conf}}{(\code{integer(1)})\cr
Parameter that affects the width of SWAP confidence intervals, default 10000}

\item{\code{loss_fn}}{(\code{character(1)})\cr
The loss function, "lp" (for p integer > 0) or one of "manhattan", "cosine", "inf" or "euclidean"}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-KMedoids-new}{\code{KMedoids$new()}}
\item \href{#method-KMedoids-get_algorithm}{\code{KMedoids$get_algorithm()}}
\item \href{#method-KMedoids-fit}{\code{KMedoids$fit()}}
\item \href{#method-KMedoids-get_medoids_final}{\code{KMedoids$get_medoids_final()}}
\item \href{#method-KMedoids-get_labels}{\code{KMedoids$get_labels()}}
\item \href{#method-KMedoids-get_statistic}{\code{KMedoids$get_statistic()}}
\item \href{#method-KMedoids-get_parallelize}{\code{KMedoids$get_parallelize()}}
\item \href{#method-KMedoids-print}{\code{KMedoids$print()}}
\item \href{#method-KMedoids-clone}{\code{KMedoids$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-KMedoids-new"></a>}}
\if{latex}{\out{\hypertarget{method-KMedoids-new}{}}}
\subsection{Method \code{new()}}{
Create a new KMedoids object
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{KMedoids$new(
  k = 5L,
  algorithm = c("BanditPAM", "PAM", "FastPAM1"),
  max_iter = 1000L,
  build_conf = 1000,
  swap_conf = 10000L,
  parallelize = TRUE
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{k}}{number of medoids/clusters to create, default 5}

\item{\code{algorithm}}{the algorithm to use, one of "BanditPAM", "PAM", "FastPAM1"}

\item{\code{max_iter}}{the maximum number of SWAP steps the algorithm runs, default 1000}

\item{\code{build_conf}}{parameter that affects the width of BUILD confidence intervals, default 1000}

\item{\code{swap_conf}}{parameter that affects the width of SWAP confidence intervals, default 10000}

\item{\code{parallelize}}{use parallelization}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
a KMedoids object which can be used to fit the banditpam algorithm to data
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-KMedoids-get_algorithm"></a>}}
\if{latex}{\out{\hypertarget{method-KMedoids-get_algorithm}{}}}
\subsection{Method \code{get_algorithm()}}{
Return the algorithm used
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{KMedoids$get_algorithm()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
a string indicating the algorithm
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-KMedoids-fit"></a>}}
\if{latex}{\out{\hypertarget{method-KMedoids-fit}{}}}
\subsection{Method \code{fit()}}{
Fit the KMedoids algorthm given the data and loss. It is advisable to set the seed before calling this method for reproducible results.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{KMedoids$fit(data, loss, dist_mat = NULL)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{data}}{the data matrix}

\item{\code{loss}}{the loss function, either "lp" (p, integer indicating L_p loss) or one of "manhattan", "cosine", "inf" or "euclidean"}

\item{\code{dist_mat}}{an optional distance matrix}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-KMedoids-get_medoids_final"></a>}}
\if{latex}{\out{\hypertarget{method-KMedoids-get_medoids_final}{}}}
\subsection{Method \code{get_medoids_final()}}{
Return the final medoid indices after clustering
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{KMedoids$get_medoids_final()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
a vector indices of the final mediods
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-KMedoids-get_labels"></a>}}
\if{latex}{\out{\hypertarget{method-KMedoids-get_labels}{}}}
\subsection{Method \code{get_labels()}}{
Return the cluster labels after clustering
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{KMedoids$get_labels()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
a vector of the cluster labels for the observations
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-KMedoids-get_statistic"></a>}}
\if{latex}{\out{\hypertarget{method-KMedoids-get_statistic}{}}}
\subsection{Method \code{get_statistic()}}{
Get the specified statistic after clustering
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{KMedoids$get_statistic(what)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{what}}{a string which should one of \code{"dist_computations"}, \code{"dist_computations_and_misc"},
\code{"misc_dist"}, \code{"build_dist"}, \code{"swap_dist"}, \code{"cache_writes"}, \code{"cache_hits"},
or \code{"cache_misses"}}

\item{\code{return}}{the statistic}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-KMedoids-get_parallelize"></a>}}
\if{latex}{\out{\hypertarget{method-KMedoids-get_parallelize}{}}}
\subsection{Method \code{get_parallelize()}}{
Return if parallelize is enabled
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{KMedoids$get_parallelize()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
true or false
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-KMedoids-print"></a>}}
\if{latex}{\out{\hypertarget{method-KMedoids-print}{}}}
\subsection{Method \code{print()}}{
Printer.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{KMedoids$print(...)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{...}}{(ignored).}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-KMedoids-clone"></a>}}
\if{latex}{\out{\hypertarget{method-KMedoids-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{KMedoids$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
