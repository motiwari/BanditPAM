import numpy as np

# These are used to plot the reference lines for scaling of PAM and FastPAM1
# in the paper. These values are computed for an example data subset using the
# ELKI implementation available at https://elki-project.github.io/

MNIST_L2_k5_P_baseline_icpt = np.log10(12.4402)
MNIST_L2_k5_FP1_baseline_icpt = np.log10((89.535+5.222)/5)
MNIST_L2_k5_x_0 = 4 # N = 10,000, log(N) = 4

MNIST_L2_k10_P_baseline_icpt = np.log10((150.157+4.813)/7)
MNIST_L2_k10_FP1_baseline_icpt = np.log10((75.106+5.007)/7)
MNIST_L2_k10_x_0 = 4 # N = 10,000, log(N) = 4

MNIST_cosine_k5_P_baseline_icpt = np.log10((81.291+4.950)/5)
MNIST_cosine_k5_FP1_baseline_icpt = np.log10((66.858+5.019)/5)
MNIST_cosine_k5_x_0 = 4 # N = 10,000, log(N) = 4

scRNA_L1_k5_P_baseline_icpt = np.log10((7.707+5.811)/2)
scRNA_L1_k5_FP1_baseline_icpt = np.log10((6.215+7.839)/2)
scRNA_L1_k5_x_0 = 3 # N = 1,000, log(N) = 3
