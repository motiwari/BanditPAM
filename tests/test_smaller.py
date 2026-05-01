import math
import unittest
import pandas as pd
import numpy as np
import sys

from banditpam import KMedoids
from utils import bpam_agrees_pam
from constants import (
    NUM_SMALL_CASES,
    SMALL_K_SCHEDULE,
    N_SMALL_K,
    SMALL_SAMPLE_SIZE,
    PROPORTION_PASSING,
)


class SmallerTests(unittest.TestCase):
    small_mnist = pd.read_csv("data/MNIST_100.csv", header=None).to_numpy()
    mnist_70k = pd.read_csv("data/MNIST_70k.csv", sep=" ", header=None)
    if sys.platform == "win32":
        scrna = pd.read_csv(
            "data/scrna_reformat.csv.gz", header=None, dtype="float16"
        )  # float16 for less memory usage
    else:
        scrna = pd.read_csv("data/scrna_reformat.csv.gz", header=None)

    def test_small_mnist(self):
        """
        Test NUM_SMALL_CASES number of test cases with subsets of size
        SMALL_SAMPLE_SIZE randomly drawn from the full MNIST dataset.
        """
        for i in range(NUM_SMALL_CASES):
            data = self.mnist_70k.sample(n=SMALL_SAMPLE_SIZE).to_numpy()

            # Test agreement with FastPAM1
            _ = bpam_agrees_pam(
                k=SMALL_K_SCHEDULE[i % N_SMALL_K],
                data=data,
                loss="L2",
                test_build=True,
                assert_immediately=True,
                use_fp=True,
            )

            # Test agreement with PAM
            _ = bpam_agrees_pam(
                k=SMALL_K_SCHEDULE[i % N_SMALL_K],
                data=data,
                loss="L2",
                test_build=True,
                assert_immediately=True,
                use_fp=False,
            )

    def test_small_scrna(self):
        """
        Test NUM_SMALL_CASES number of test cases with subsets of size
        SMALL_SAMPLE_SIZE randomly drawn from the full scRNA dataset.
        """
        count = 0
        for i in range(NUM_SMALL_CASES):
            data = self.scrna.sample(n=SMALL_SAMPLE_SIZE).to_numpy()

            # Test agreement with FastPAM1
            count += bpam_agrees_pam(
                k=SMALL_K_SCHEDULE[i % N_SMALL_K],
                data=data,
                loss="L1",
                test_build=True,
                assert_immediately=False,
                use_fp=True,
            )

            # Test agreement with PAM
            count += bpam_agrees_pam(
                k=SMALL_K_SCHEDULE[i % N_SMALL_K],
                data=data,
                loss="L1",
                test_build=True,
                assert_immediately=False,
                use_fp=False,
            )
        # Occasionally some may fail due to degeneracy in the scRNA dataset
        self.assertTrue(count >= 2 * PROPORTION_PASSING * NUM_SMALL_CASES)

    def test_small_mnist_known_cases(self):
        """
        Test BanditPAM on a subset of MNIST with known solutions for both k = 5
        and k = 10, after both the BUILD and SWAP steps.
        """
        kmed_5 = KMedoids(
            n_medoids=5,
            algorithm="BanditPAM",
        )
        kmed_5.seed = 0
        kmed_5.fit(self.small_mnist, "L2")

        self.assertEqual(
            sorted(kmed_5.build_medoids.tolist()), [16, 24, 32, 70, 87]
        )
        self.assertEqual(sorted(kmed_5.medoids.tolist()), [23, 30, 49, 70, 99])

        kmed_10 = KMedoids(
            n_medoids=10,
            algorithm="BanditPAM",
        )
        kmed_10.seed = 0
        kmed_10.fit(self.small_mnist, "L2")

        self.assertEqual(
            sorted(kmed_10.build_medoids.tolist()),
            [16, 24, 32, 49, 70, 82, 87, 90, 94, 99],
        )
        self.assertEqual(
            sorted(kmed_10.medoids.tolist()),
            [16, 25, 31, 49, 63, 70, 82, 90, 94, 99],
        )

    def test_edge_cases(self):
        """
        Test that BanditPAM raises errors on n_medoids being unspecified or an
        empty dataset.
        """
        kmed = KMedoids()
        kmed.seed = 0

        # initialized to empty
        self.assertEqual([], kmed.build_medoids[0].tolist())
        self.assertEqual([], kmed.medoids[0].tolist())

        # error that no value of k has been passed
        self.assertRaises(ValueError, kmed.fit, np.array([]), "L2")

        # error on trying to fit on empy
        self.assertRaises(ValueError, kmed.fit, np.array([]), "L2")

    def test_swap_timing_stats(self):
        """
        After fit(), total_swap_time (ms) and time_per_swap are consistent.

        For BanditPAM, BanditPAM_orig, and PAM, each SWAP outer iteration is
        timed once and matches ``steps``, so time_per_swap * steps equals
        total_swap_time (ms).

        FastPAM1 counts ``steps`` as outer refinement iterations while timing
        multiple inner swaps per call; we only assert types and non-negativity.
        """
        data = self.small_mnist
        for algo in ("BanditPAM", "BanditPAM_orig", "FastPAM1", "PAM"):
            with self.subTest(algorithm=algo):
                kmed = KMedoids(n_medoids=5, algorithm=algo, max_iter=100)
                kmed.seed = 0
                kmed.fit(data, "L2")
                total_ms = kmed.total_swap_time
                tps = kmed.time_per_swap
                n_swaps = kmed.steps
                self.assertIsInstance(total_ms, int)
                self.assertIsInstance(tps, float)
                self.assertGreaterEqual(total_ms, 0)
                self.assertGreaterEqual(tps, 0.0)
                if total_ms == 0:
                    self.assertEqual(tps, 0.0)

                if algo == "FastPAM1":
                    continue

                if n_swaps == 0:
                    self.assertEqual(total_ms, 0)
                    self.assertEqual(tps, 0.0)
                    continue

                reconstructed = tps * float(n_swaps)
                ok = math.isclose(
                    reconstructed,
                    float(total_ms),
                    rel_tol=1e-5,
                    abs_tol=1e-4,
                )
                self.assertTrue(
                    ok,
                    msg=(
                        f"{algo}: time_per_swap * steps = {reconstructed}, "
                        f"total_swap_time = {total_ms}, steps = {n_swaps}"
                    ),
                )


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
