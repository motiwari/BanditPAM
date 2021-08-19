import unittest
from BanditPAM import KMedoids
import pandas as pd
import numpy as np


def onFly(k, data, loss):
    kmed_bpam = KMedoids(n_medoids=k, algorithm="BanditPAM")
    kmed_naive = KMedoids(n_medoids=k, algorithm="naive")
    kmed_bpam.fit(data, loss)
    kmed_naive.fit(data, loss)

    if (kmed_bpam.medoids.tolist() == kmed_naive.medoids.tolist()) and (
        kmed_bpam.build_medoids.tolist() == kmed_naive.build_medoids.tolist()
    ):
        return 1
    else:
        return 0


class PythonTests(unittest.TestCase):
    small_mnist = pd.read_csv("./data/MNIST.csv", header=None).to_numpy()

    mnist_70k = pd.read_csv("./data/MNIST-70k.csv", sep=" ", header=None)

    scrna = pd.read_csv("./data/scrna_reformat.csv.gz", header=None)

    def test_small_on_fly_mnist(self):
        """
        Test 10 on-the-fly generated samples of 1000 datapoints from mnist-70k dataset
        Must get correct medoids on roughly 95% of cases
        """
        count = 0
        k_schedule = [4, 6, 8, 10] * 12
        for i in range(48):
            data = self.mnist_70k.sample(n=1000).to_numpy()
            count += onFly(k=k_schedule[i], data=data, loss="L2")
        self.assertTrue(count >= 45)

    def test_small_on_fly_scrna(self):
        """
        Test 10 on-the-fly generated samples of 1000 datapoints from scrna dataset
        Must get correct medoids on roughly 95% of cases
        """
        count = 0
        k_schedule = [4, 6, 8, 10] * 12
        for i in range(48):
            data = self.scrna.sample(n=1000).to_numpy()
            count += onFly(k=k_schedule[i], data=data, loss="L1")
        self.assertTrue(count >= 45)

    def test_large_on_fly_mnist(self):
        """
        Test 10 on-the-fly generated samples of 1000 datapoints from mnist-70k dataset
        Must get correct medoids on roughly 95% of cases
        """
        count = 0
        k_schedule = [4, 6, 8, 10] * 12
        size_schedule = [1000, 2000, 3000, 4000, 5000] * 10
        for i in range(48):
            data = mnist_70k.sample(n=size_schedule[i])
            count += onFly(k=k_schedule[i], data=data, loss="L2")
        self.assertTrue(count >= 45)

    def test_large_on_fly_scrna(self):
        """
        Test 10 on-the-fly generated samples of 1000 datapoints from scrna dataset
        Must get correct medoids on roughly 95% of cases
        """
        count = 0
        k_schedule = [4, 6, 8, 10] * 12
        size_schedule = [1000, 2000, 3000, 4000, 5000] * 10
        for i in range(48):
            data = scrna.sample(n=size_schedule[i])
            count += onFly(k=k_schedule[i], data=data, loss="L2")
        self.assertTrue(count >= 45)

    def test_time_cases_mnist(self):
        """
        Evaluate the n*log(n) scaling of the BanditPAM algorithm on MNIST datasets.
        """
        MNIST_10k = mnist_70k.head(10000).to_numpy()
        kmed1 = KMedoids(n_medoids=5, algorithm="BanditPAM", verbosity=0)
        kmed1.fit(MNIST_10k, "L2")

        MNIST_schedule = [20, 40, 70]
        kmed2 = KMedoids(n_medoids=5, algorithm="BanditPAM", verbosity=0)
        for num in MNIST_schedule:
            MNIST_test = mnist_70k.head(num * 1000).to_numpy()
            kmed2.fit(MNIST_test, "L2")
            self.assertTrue(kmed2.steps < ((num / 10) ** 1.2) * kmed1.steps)

    def test_time_cases_scrna(self):
        """
        Evaluate the n*log(n) scaling of the BanditPAM algorithm on scRNA datasets.
        """
        SCRNA_10k = scrna.head(10000).to_numpy()
        kmed1 = KMedoids(n_medoids=5, algorithm="BanditPAM", verbosity=0)
        kmed1.fit(SCRNA_10k, "L1")

        scrna_schedule = [20, 30, 40]
        kmed2 = KMedoids(n_medoids=5, algorithm="BanditPAM", verbosity=0)
        for num in scrna_schedule:
            scrna_test = scrna.head(num * 1000).to_numpy()
            kmed2.fit(MNIST_test, "L1")
            self.assertTrue(kmed2.steps < ((num / 10) ** 1.2) * kmed1.steps)


if __name__ == "__main__":
    unittest.main()
