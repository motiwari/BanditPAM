.onAttach <- function(libname, pkgname) {
  if (!interactive()) return()
  n_threads <- bpam_num_threads()
  if (n_threads == 1) {
    packageStartupMessage("BanditPAM: OpenMP not in effect!")
  } else {
    ## This erroneously reports hyperthreads too
    packageStartupMessage(sprintf("BanditPAM: using %d threads", n_threads))
  }
}
