
## Constants
library(banditpam)

NUM_SMALL_CASES <- 10
NUM_MEDIUM_CASES <- 48

SMALL_K_SCHEDULE <- c(4, 6, 8, 10)
N_SMALL_K <- length(SMALL_K_SCHEDULE)

SMALL_SAMPLE_SIZE <- 100
MEDIUM_SAMPLE_SIZE <- 1000
LARGE_SAMPLE_SIZE <- 10000

MEDIUM_SIZE_SCHEDULE <- c(1000, 2000, 3000, 4000, 5000)
NUM_MEDIUM_SIZES <- length(MEDIUM_SIZE_SCHEDULE)

MNIST_SIZE_MULTIPLIERS <- c(2, 4, 7)
SCRNA_SIZE_MULTIPLIERS <- c(2, 3, 4)

PROPORTION_PASSING <- 0.8
SCALING_EXPONENT <- 1.2

MILLISECONDS_IN_A_SECOND <- 1000

small_data <- readRDS("mnist_1k.RDS")

## For reproducibility
set.seed(123)

for (k in SMALL_K_SCHEDULE) {
  kmed_bpam <- KMedoids$new(k = k, algorithm = "BanditPAM")
  kmed_pam <- KMedoids$new(k = k, algorithm = "FastPAM1")
  kmed_bpam$fit(small_data, loss = "l2")
  kmed_pam$fit(small_data, loss = "l2")
  
  bpam_final_medoids = sort(kmed_bpam$get_medoids_final())
  pam_final_medoids = sort(kmed_pam$get_medoids_final())
  expect_equal(bpam_final_medoids, pam_final_medoids)
}
