import unittest
import pandas as pd
import numpy as np
import time

from banditpam import KMedoids
from utils import bpam_agrees_pam
from constants import (
    NUM_MEDIUM_CASES,
    SMALL_K_SCHEDULE,
    N_SMALL_K,
    MEDIUM_SAMPLE_SIZE,
    LARGE_SAMPLE_SIZE,
    MEDIUM_SIZE_SCHEDULE,
    NUM_MEDIUM_SIZES,
    MNIST_SIZE_MULTIPLIERS,
    SCRNA_SIZE_MULTIPLIERS,
    PROPORTION_PASSING,
    SCALING_EXPONENT,
)


class LargerTests(unittest.TestCase):
    mnist_70k = pd.read_csv("data/MNIST_70k.csv", sep=" ", header=None)
    scrna = pd.read_csv("data/scrna_reformat.csv.gz", header=None)

    def test_medium_mnist(self):
        """
        Generate NUM_MEDIUM_CASES random subsets of MNIST of size
        MEDIUM_SAMPLE_SIZE and verify BanditPAM agrees with PAM.
        """
        num_succeed = 0
        for i in range(NUM_MEDIUM_CASES):
            data = self.mnist_70k.sample(n=MEDIUM_SAMPLE_SIZE).to_numpy()
            num_succeed += bpam_agrees_pam(
                k=SMALL_K_SCHEDULE[i % N_SMALL_K],
                data=data,
                loss="L2",
                test_build=True,
                assert_immediately=False,
            )
        self.assertTrue(
            num_succeed >= PROPORTION_PASSING * NUM_MEDIUM_CASES
        )  # avoids stochasticity issues

    def test_various_medium_mnist(self):
        """
        Generate NUM_MEDIUM_CASES random subsets of MNIST of various sizes and
        verify BanditPAM agrees with PAM.

        Since PAM is very slow, we can only do this for fairly small sizes in
        MEDIUM_SIZE_SCHEDULE
        """
        num_succeed = 0
        for i in range(NUM_MEDIUM_CASES):
            size = MEDIUM_SIZE_SCHEDULE[i % NUM_MEDIUM_SIZES]
            data = self.mnist_70k.sample(n=size)
            num_succeed += bpam_agrees_pam(
                k=SMALL_K_SCHEDULE[i % N_SMALL_K],
                data=data,
                loss="L2",
                test_build=True,
                assert_immediately=False,
            )
        self.assertTrue(
            num_succeed >= PROPORTION_PASSING * NUM_MEDIUM_CASES
        )  # avoids stochasticity issues

    def test_time_cases_mnist(self):
        """
        Verify that BanditPAM scales as O(nlogn) on the MNIST dataset.
        """
        MNIST_10k = self.mnist_70k.head(LARGE_SAMPLE_SIZE).to_numpy()
        kmed = KMedoids(n_medoids=5, algorithm="BanditPAM")
        kmed.seed = 0
        start = time.time()
        kmed.fit(MNIST_10k, "L2")
        base_runtime = time.time() - start

        for size_multiplier in MNIST_SIZE_MULTIPLIERS:
            size = size_multiplier * LARGE_SAMPLE_SIZE
            MNIST_test = self.mnist_70k.head(n=size).to_numpy()
            start = time.time()
            kmed.fit(MNIST_test, "L2")
            runtime = time.time() - start
            # TODO(@motiwari): Timing test will not work
            # need to compute it over number of steps
            self.assertTrue(
                runtime < (size_multiplier**SCALING_EXPONENT) * base_runtime
            )

    def test_medium_scrna(self):
        """
        Generate NUM_MEDIUM_CASES random subsets of MNIST of size
        MEDIUM_SAMPLE_SIZE and verify BanditPAM agrees with PAM.
        """
        count = 0
        for i in range(NUM_MEDIUM_CASES):
            data = self.scrna.sample(n=MEDIUM_SAMPLE_SIZE).to_numpy()
            count += bpam_agrees_pam(
                k=SMALL_K_SCHEDULE[i % N_SMALL_K],
                data=data,
                loss="L1",
                test_build=False,
                assert_immediately=False,
            )
        self.assertTrue(count >= PROPORTION_PASSING * NUM_MEDIUM_CASES)

    def test_various_medium_scrna(self):
        """
        Generate NUM_MEDIUM_CASES random subsets of scRNA of various sizes and
        verify BanditPAM agrees with PAM.

        Since PAM is very slow, we can only do this for fairly small sizes in
        MEDIUM_SIZE_SCHEDULE
        """
        count = 0
        for i in range(NUM_MEDIUM_CASES):
            size = MEDIUM_SIZE_SCHEDULE[i % NUM_MEDIUM_SIZES]
            data = self.scrna.sample(n=size)
            count += bpam_agrees_pam(
                k=SMALL_K_SCHEDULE[i % N_SMALL_K],
                data=data,
                loss="L2",
                test_build=False,
                assert_immediately=False,
            )
        self.assertTrue(count >= PROPORTION_PASSING * NUM_MEDIUM_CASES)

    def test_time_cases_scrna(self):
        """
        Verify that BanditPAM scales as O(nlogn) on the MNIST dataset.
        """
        SCRNA_10k = self.scrna.head(LARGE_SAMPLE_SIZE).to_numpy()
        kmed = KMedoids(n_medoids=5, algorithm="BanditPAM")
        kmed.seed = 0
        start = time.time()
        kmed.fit(SCRNA_10k, "L1")
        base_runtime = time.time() - start

        for size_multiplier in SCRNA_SIZE_MULTIPLIERS:
            size = size_multiplier * LARGE_SAMPLE_SIZE
            scrna_test = self.scrna.head(n=size).to_numpy()
            start = time.time()
            kmed.fit(scrna_test, "L1")
            runtime = time.time() - start
            # TODO(@motiwari): Timing test will not work
            # need to compute it over number of steps
            self.assertTrue(
                runtime < (size_multiplier**SCALING_EXPONENT) * base_runtime
            )


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
