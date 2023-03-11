.onAttach <- function(libname, pkgname) {
  if (!interactive()) return()
  n_threads <- bpam_num_threads()
  if (n_threads == 1) {
    packageStartupMessage("banditpam: OpenMP not in effect!")
  } else {
    ## This erroneously reports hyperthreads too
    packageStartupMessage(sprintf("banditpam: using %d (hyper) threads.", n_threads))
  }
}
