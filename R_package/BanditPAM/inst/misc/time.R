library(BanditPAM)
library(MASS)
# Generate data from a Gaussian Mixture Model with the given means:
set.seed(10)

## MNIST
k <- 10
d <- as.matrix(data.table::fread("~/tmp/BanditPAM/data/MNIST_1k.csv"))
obj <- KMedoids$new(k = k)
system.time(obj$fit(data = d, loss = "l2"))

library(profvis)
profvis(obj$fit(data = d, loss = "l2"))

meds <- obj$get_medoids_final()

system.time(obj$fit(data = d, loss = "l2"))
