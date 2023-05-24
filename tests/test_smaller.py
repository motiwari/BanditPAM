import unittest
import pandas as pd
import numpy as np

from banditpam import KMedoids
from utils import bpam_agrees_pam
from constants import (
    NUM_SMALL_CASES,
    SMALL_K_SCHEDULE,
    N_SMALL_K,
    SMALL_SAMPLE_SIZE,
    PROPORTION_PASSING,
)
from memory_profiler import profile


# TODO(@motiwari): Set seeds
class SmallerTests(unittest.TestCase):
    small_mnist = pd.read_csv("data/MNIST_100.csv", header=None).to_numpy()

    @profile
    def test_small_mnist_known_cases(self):
        """
        Test BanditPAM on a subset of MNIST with known solutions
        for both k = 5 and k = 10, after both the BUILD and SWAP steps
        """
        kmed_5 = KMedoids(
            n_medoids=5,
            algorithm="BanditPAM",
        )
        kmed_5.fit(self.small_mnist, "L2")

        self.assertEqual(
            sorted(kmed_5.build_medoids.tolist()),
            [16, 24, 32, 70, 87]
        )
        self.assertEqual(
            sorted(kmed_5.medoids.tolist()),
            [23, 30, 49, 70, 99]
        )

        kmed_10 = KMedoids(
            n_medoids=10,
            algorithm="BanditPAM",
        )
        kmed_10.fit(self.small_mnist, "L2")

        self.assertEqual(
            sorted(kmed_10.build_medoids.tolist()),
            [16, 24, 32, 49, 70, 82, 87, 90, 94, 99],
        )
        self.assertEqual(
            sorted(kmed_10.medoids.tolist()),
            [16, 25, 31, 49, 63, 70, 82, 90, 94, 99]
        )

if __name__ == "__main__":
    unittest.main()
