import unittest
import pandas as pd
import numpy as np

from banditpam import KMedoids
from utils import bpam_agrees_pam
from constants import *


class LargerTests(unittest.TestCase):
    mnist_70k = pd.read_csv("data/MNIST_70k.csv", sep=" ", header=None)
    scrna = pd.read_csv("data/scrna_reformat.csv.gz", header=None)

    def test_medium_mnist(self):
        """
        Generate NUM_MEDIUM_CASES random subsets of MNIST 
        of size MEDIUM_SAMPLE_SIZE and verify BanditPAM agrees with PAM
        """
        for i in range(NUM_MEDIUM_CASES):
            data = self.mnist_70k.sample(n=MEDIUM_SAMPLE_SIZE).to_numpy()
            _ = bpam_agrees_pam(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L2", test_build=True, assert_immediately=True)

    def test_various_medium_mnist(self):
        """
        Generate NUM_MEDIUM_CASES random subsets of MNIST of various sizes
        and verify BanditPAM agrees with PAM. Since PAM is very slow,
        we can only do this for fairly small sizes in MEDIUM_SIZE_SCHEDULE
        """
        for i in range(NUM_MEDIUM_CASES):
            data = self.mnist_70k.sample(n=MEDIUM_SIZE_SCHEDULE[i % NUM_MEDIUM_SIZES])
            _ = bpam_agrees_pam(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L2", test_build=True, assert_immediately=True)

    def test_time_cases_mnist(self):
        """
        Verify that BanditPAM scales as O(nlogn) on the MNIST dataset.
        """
        MNIST_10k = self.mnist_70k.head(LARGE_SAMPLE_SIZE).to_numpy()
        kmed1 = KMedoids(n_medoids=5, algorithm="BanditPAM")
        kmed1.fit(MNIST_10k, "L2")

        kmed2 = KMedoids(n_medoids=5, algorithm="BanditPAM")
        for size_multiplier in MNIST_SIZE_MULTIPLIERS:
            MNIST_test = self.mnist_70k.head(size_multiplier * LARGE_SAMPLE_SIZE).to_numpy()
            kmed2.fit(MNIST_test, "L2")
            self.assertTrue(kmed2.steps < (size_multiplier ** SCALING_EXPONENT) * kmed1.steps)

    def test_medium_scrna(self):
        """
        Generate NUM_MEDIUM_CASES random subsets of MNIST 
        of size MEDIUM_SAMPLE_SIZE and verify BanditPAM agrees with PAM
        """
        count = 0
        for i in range(NUM_MEDIUM_CASES):
            data = self.scrna.sample(n=MEDIUM_SAMPLE_SIZE).to_numpy()
            count += bpam_agrees_pam(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L1", test_build=False, assert_immediately=False)
        self.assertTrue(count >= PROPORTION_PASSING*NUM_MEDIUM_CASES)

    def test_various_medium_scrna(self):
        """
        Generate NUM_MEDIUM_CASES random subsets of scRNA of various sizes
        and verify BanditPAM agrees with PAM. Since PAM is very slow,
        we can only do this for fairly small sizes in MEDIUM_SIZE_SCHEDULE
        """
        count = 0
        for i in range(NUM_MEDIUM_CASES):
            data = self.scrna.sample(n=MEDIUM_SIZE_SCHEDULE[i % NUM_MEDIUM_SIZES])
            count += bpam_agrees_pam(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L2", test_build=False, assert_immediately=False)
        self.assertTrue(count >= PROPORTION_PASSING*NUM_MEDIUM_CASES)

    def test_time_cases_scrna(self):
        """
        Verify that BanditPAM scales as O(nlogn) on the MNIST dataset.
        """
        SCRNA_10k = self.scrna.head(LARGE_SAMPLE_SIZE).to_numpy()
        kmed1 = KMedoids(n_medoids=5, algorithm="BanditPAM")
        kmed1.fit(SCRNA_10k, "L1")

        kmed2 = KMedoids(n_medoids=5, algorithm="BanditPAM")
        for size_multiplier in SCRNA_SIZE_MULTIPLIERS:
            scrna_test = self.scrna.head(size_multiplier * LARGE_SAMPLE_SIZE).to_numpy()
            kmed2.fit(scrna_test, "L1")
            self.assertTrue(kmed2.steps < (size_multiplier ** SCALING_EXPONENT) * kmed1.steps)

if __name__ == "__main__":
    unittest.main()
