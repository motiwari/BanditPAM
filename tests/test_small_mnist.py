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


# TODO(@motiwari): Set seeds
class SmallMnistTests(unittest.TestCase):
    small_mnist = pd.read_csv("data/MNIST_100.csv", header=None).to_numpy()

    def test_small_mnist(self):
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

        kmed_5_cache = KMedoids(
            n_medoids=5,
            algorithm="BanditPAM",
            useCacheP=True,
        )
        kmed_5_cache.fit(self.small_mnist, "L2")

        self.assertEqual(
            sorted(kmed_5_cache.build_medoids.tolist()),
            [16, 24, 32, 70, 87]
        )
        self.assertEqual(
            sorted(kmed_5_cache.medoids.tolist()),
            [23, 30, 49, 70, 99]
        )

        kmed_5_cache_perm = KMedoids(
            n_medoids=5,
            algorithm="BanditPAM",
            useCacheP=True,
            usePerm=True,
        )
        kmed_5_cache_perm.fit(self.small_mnist, "L2")

        self.assertEqual(
            sorted(kmed_5_cache_perm.build_medoids.tolist()),
            [16, 24, 32, 70, 87]
        )
        self.assertEqual(
            sorted(kmed_5_cache_perm.medoids.tolist()),
            [23, 30, 49, 70, 99]
        )

        kmed_10 = KMedoids(
            n_medoids=10,
            algorithm="BanditPAM",
        )
        kmed_10.fit(self.small_mnist, "L2")

        self.assertEqual(
            sorted(kmed_10.build_medoids.tolist()),
            [16, 24, 32, 49, 70, 82, 87, 90, 94, 99]
        )
        self.assertEqual(
            sorted(kmed_10.medoids.tolist()),
            [16, 25, 31, 49, 63, 70, 82, 90, 94, 99]
        )

    def test_edge_cases(self):
        """
        Test that BanditPAM raises errors on n_medoids being unspecified or
        an empty dataset
        """
        kmed = KMedoids()

        # initialized to empty
        self.assertEqual([], kmed.build_medoids[0].tolist())
        self.assertEqual([], kmed.medoids[0].tolist())

        # error that no value of k has been passed
        self.assertRaises(ValueError, kmed.fit, np.array([]), "L2")

        # error on trying to fit on empy
        self.assertRaises(ValueError, kmed.fit, np.array([]), "L2")


if __name__ == "__main__":
    unittest.main()
