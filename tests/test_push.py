import unittest
import pandas as pd
import numpy as np

from banditpam import KMedoids
from utils import bpam_agrees_pam
from constants import *


class PythonTests(unittest.TestCase):
    mnist_70k = pd.read_csv("data/MNIST_70k.csv", sep=" ", header=None)
    scrna = pd.read_csv("data/scrna_reformat.csv.gz", header=None)

    def test_small_on_the_fly_mnist(self):
        """
        Generate NUM_LARGE_CASES random subsets of MNIST 
        of size MEDIUM_SAMPLE_SIZE and verify BanditPAM agrees with PAM
        """
        count = 0
        for i in range(NUM_LARGE_CASES):
            data = self.mnist_70k.sample(n=MEDIUM_SAMPLE_SIZE).to_numpy()
            count += bpam_agrees_pam(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L2")
        self.assertTrue(count == NUM_LARGE_CASES)

    def test_large_on_the_fly_mnist(self):
        """
        Generate NUM_LARGE_CASES random subsets of MNIST various sizes
        and verify BanditPAM agrees with PAM. Since PAM is very slow,
        we can only do this for fairly small sizes in MEDIUM_SIZE_SCHEDULE
        """
        count = 0
        for i in range(NUM_LARGE_CASES):
            data = self.mnist_70k.sample(n=MEDIUM_SIZE_SCHEDULE[i % NUM_MEDIUM_SIZES])
            count += bpam_agrees_pam(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L2")
        self.assertTrue(count == NUM_LARGE_CASES)

    def test_time_cases_mnist(self):
        """
        Evaluate the n*log(n) scaling of the BanditPAM algorithm on MNIST datasets.
        """
        MNIST_10k = self.mnist_70k.head(LARGE_SAMPLE_SIZE).to_numpy()
        kmed1 = KMedoids(n_medoids=5, algorithm="BanditPAM")
        kmed1.fit(MNIST_10k, "L2")

        kmed2 = KMedoids(n_medoids=5, algorithm="BanditPAM")
        for num in MNIST_SIZE_MULTIPLIERS:
            MNIST_test = self.mnist_70k.head(num * LARGE_SAMPLE_SIZE).to_numpy()
            kmed2.fit(MNIST_test, "L2")
            self.assertTrue(kmed2.steps < (num ** SCALING_EXPONENT) * kmed1.steps)

    def test_small_on_the_fly_scrna(self):
        """
        Test 10 on-the-fly generated samples of 1000 datapoints from scrna dataset
        Must get correct medoids on roughly 95% of cases
        """
        count = 0
        for i in range(NUM_LARGE_CASES):
            data = self.scrna.sample(n=MEDIUM_SAMPLE_SIZE).to_numpy()
            count += bpam_agrees_pam(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L1")
        self.assertTrue(count >= PROPORTION_PASSING*NUM_LARGE_CASES)

    def test_large_on_the_fly_scrna(self):
        """
        Test 10 on-the-fly generated samples of 1000 datapoints from scrna dataset
        Must get correct medoids on roughly 95% of cases
        """
        count = 0
        for i in range(NUM_LARGE_CASES):
            data = self.scrna.sample(n=MEDIUM_SIZE_SCHEDULE[i % NUM_MEDIUM_SIZES])
            count += bpam_agrees_pam(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L2")
        self.assertTrue(count >= PROPORTION_PASSING*NUM_LARGE_CASES)

    def test_time_cases_scrna(self):
        """
        Evaluate the n*log(n) scaling of the BanditPAM algorithm on scRNA datasets.
        """
        SCRNA_10k = self.scrna.head(LARGE_SAMPLE_SIZE).to_numpy()
        kmed1 = KMedoids(n_medoids=5, algorithm="BanditPAM")
        kmed1.fit(SCRNA_10k, "L1")

        kmed2 = KMedoids(n_medoids=5, algorithm="BanditPAM")
        for num in SCRNA_SIZE_MULTIPLIERS:
            scrna_test = self.scrna.head(num * LARGE_SAMPLE_SIZE).to_numpy()
            kmed2.fit(scrna_test, "L1")
            self.assertTrue(kmed2.steps < (num ** SCALING_EXPONENT) * kmed1.steps)

if __name__ == "__main__":
    unittest.main()
