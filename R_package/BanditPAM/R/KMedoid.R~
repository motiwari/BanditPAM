#' @title KMedoids Class
#'
#' @description
#' This class wraps around the C++ KMedoids class and exposes methods and fields of the C++ object.
#'
#' @examples
#' # Generate data from a Gaussian Mixture Model with the given means:
#' set.seed(10)
#' n_per_cluster <- 40
#' means <- list(c(0, 0), c(-5, 5), c(5, 5))
#' X <- do.call(rbind, lapply(means, MASS::mvrnorm, n = n_per_cluster, Sigma = diag(2)))
#' obj <- KMedoids$new(k = 3)
#' obj$fit(data = X, loss = "l2")
#' meds <- obj$get_medoids_final()
#' plot(X[, 1], X[, 2])
#' points(X[meds, 1], X[meds, 2], col = "red", pch = 19)
#' @importFrom Rcpp evalCpp
#' @export
KMedoids <- R6::R6Class( "KMedoids"
,
  private = list(
    xptr = NA
  )
, 
  active = list(
    #' @field k (`integer(1)`)\cr
    #' The number of medoids/clusters to create
    k = function(value) {
      if (missing(value)) {
        .Call('_BanditPAM_KMedoids__get_k', PACKAGE = 'BanditPAM', private$xptr)
      } else {
        invisible(.Call('_BanditPAM_KMedoids__set_k', PACKAGE = 'BanditPAM', private$xptr, value))
      }
    }
   ,
    #' @field max_iter (`integer(1)`)\cr
    #' max_iter the maximum number of SWAP steps the algorithm runs
    max_iter = function(value) {
      if (missing(value)) {
        .Call('_BanditPAM_KMedoids__get_max_iter', PACKAGE = 'BanditPAM', private$xptr)
      } else {
        invisible(.Call('_BanditPAM_KMedoids__set_max_iter', PACKAGE = 'BanditPAM', private$xptr, value))
      }
    }
   ,
    #' @field build_conf (`integer(1)`)\cr
    #' Parameter that affects the width of BUILD confidence intervals, default 1000
    build_conf = function(value) {
      if (missing(value)) {
        .Call('_BanditPAM_KMedoids__get_build_conf', PACKAGE = 'BanditPAM', private$xptr)
      } else {
        invisible(.Call('_BanditPAM_KMedoids__set_build_conf', PACKAGE = 'BanditPAM', private$xptr, value))
      }
    }
   ,
    #' @field swap_conf (`integer(1)`)\cr
    #' Parameter that affects the width of SWAP confidence intervals, default 10000
    swap_conf = function(value) {
      if (missing(value)) {
        .Call('_BanditPAM_KMedoids__get_swap_conf', PACKAGE = 'BanditPAM', private$xptr)
      } else {
        invisible(.Call('_BanditPAM_KMedoids__set_swap_conf', PACKAGE = 'BanditPAM', private$xptr, value))
      }
    }
   ,
    #' @field loss_fn (`character(1)`)\cr
    #' The loss function, "lp" (for p integer > 0) or one of "manhattan", "cosine", "inf" or "euclidean"    
    loss_fn = function(value) {
      if (missing(value)) {
        .Call('_BanditPAM_KMedoids__get_loss_fn', PACKAGE = 'BanditPAM', private$xptr)
      } else {
        invisible(.Call('_BanditPAM_KMedoids__set_loss_fn', PACKAGE = 'BanditPAM', private$xptr, value))
      }
    }
  )
,
  public = list(
    #' @description
    #' Create a new KMedoids object
    #' @param k number of medoids/clusters to create, default 5
    #' @param max_iter the maximum number of SWAP steps the algorithm runs, default 1000
    #' @param build_conf parameter that affects the width of BUILD confidence intervals, default 1000
    #' @param swap_conf parameter that affects the width of SWAP confidence intervals, default 10000
    #' @return a KMedoids object which can be used to fit the BanditPAM algorithm to data
    initialize = function(k = 5L, max_iter = 1000L, build_conf = 1000, swap_conf = 10000L) {
      private$xptr <- .Call('_BanditPAM_KMedoids__new', PACKAGE = 'BanditPAM', k, max_iter, build_conf, swap_conf)
    }
   ,
    #' @description
    #' Fit the KMedoids algorthm given the data and loss. It is advisable to set the seed before calling this method for reproducible results.
    #' @param data the data matrix
    #' @param loss the loss function, either "lp" (p, integer indicating L_p loss) or one of "manhattan", "cosine", "inf" or "euclidean"
    fit = function(data, loss) {
      loss <- tolower(loss)
      if (!grepl("l[1-9]+$", loss)) {
        loss <- match.arg(loss, c("manhattan", "cosine", "inf", "euclidean"))
      }
      invisible(.Call('_BanditPAM_KMedoids__fit', PACKAGE = 'BanditPAM', private$xptr, data, loss))
    }
   ,
    #' @description
    #' Return the final medoid indices after clustering
    #' @param return a vector indices of the final mediods
    get_medoids_final = function() {
      .Call('_BanditPAM_KMedoids__get_medoids_final', PACKAGE = 'BanditPAM', private$xptr)
    }
   ,

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...) {
      catn(sprintf("<%s>", class(self)[1L]))
      catn(sprintf("\tk: %d", self$k))
      catn(sprintf("\tmax_iter: %d", self$max_iter))
      catn(sprintf("\tloss: %s", self$loss_fn))
      catn(sprintf("\tbuild_confidence: %d", self$build_conf))
      catn(sprintf("\tswap_confidence: %d", self$swap_conf))
    }
  )
  )


