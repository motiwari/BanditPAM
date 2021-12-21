import unittest
import pandas as pd
import numpy as np

from banditpam import KMedoids
from utils import SMALL_K_SCHEDULE, N_SMALL_K, NUM_SMALL_CASES, SMALL_SAMPLE_SIZE, PROPORTION_PASSING, on_the_fly

class PythonTests(unittest.TestCase):
    small_mnist = pd.read_csv("./data/MNIST.csv", header=None).to_numpy()

    mnist_70k = pd.read_csv("./data/MNIST-70k.csv", sep=" ", header=None)

    scrna = pd.read_csv("./data/scrna_reformat.csv.gz", header=None)

    def test_small_on_the_fly_mnist(self):
        """
        Test NUM_SMALL_CASES number of test cases with subsets of size 
        SMALL_SAMPLE_SIZE randomly drawn from the full MNIST dataset
        """
        count = 0
        for i in range(NUM_SMALL_CASES):
            data = self.mnist_70k.sample(n=SMALL_SAMPLE_SIZE).to_numpy()
            count += on_the_fly(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L2")
        self.assertTrue(count == NUM_SMALL_CASES) # All cases must pass

    def test_small_on_the_fly_scrna(self):
        """
        Test NUM_SMALL_CASES number of test cases with subsets of size 
        SMALL_SAMPLE_SIZE randomly drawn from the full scRNA dataset
        """
        count = 0
        for i in range(NUM_SMALL_CASES):
            data = self.scrna.sample(n=SMALL_SAMPLE_SIZE).to_numpy()
            count += on_the_fly(k=SMALL_K_SCHEDULE[i % N_SMALL_K], data=data, loss="L1")
        self.assertTrue(count >= PROPORTION_PASSING*NUM_SMALL_CASES) # Occasionally some may fail due to degeneracy in the scRNA dataset

    def test_small_mnist_cases(self):
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
