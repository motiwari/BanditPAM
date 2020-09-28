import unittest
from banditPAM import KMedoids
import pandas as pd
import numpy as np

small_mnist = pd.read_csv('./data/mnist.csv', sep=' ', header=None).to_numpy()

mnist_70k = pd.read_csv('./data/MNIST-70k.csv', sep=' ', header=None)

scrna = pd.read_csv('./data/scrna_reformat.csv.gz', header=None)

def on_the_fly(k, data, loss):
    kmed_bpam = KMedoids(k = k, algorithm = "BanditPAM")
    kmed_naive = KMedoids(k = k, algorithm = "naive")
    kmed_bpam.fit(data, loss)
    kmed_naive.fit(data, loss)
    if (kmed_bpam.final_medoids == kmed_naive.final_medoids and
        kmed_bpam.build_medoids == kmed_naive.build_medoids):
        return 1
    else:
        return 0

class PythonTests(unittest.TestCase):
    def test_large_on_fly_cases(self):
        count = 0
        k_schedule = [4, 6, 8, 10] * 25
        size_schedule = [1000, 2000, 3000, 4000, 5000] * 20
        for i in range(50):
            data = mnist_70k.sample(n = size_schedule[i])
            count += on_the_fly(k = k_schedule[i], data = data, loss = "L2")

        for i in range(50, 100):
            data = scrna.sample(n = size_schedule[i])
            count += on_the_fly(k = k_schedule[i], data = data, loss = "L1")

        self.assertTrue(count >= 95)

    def test_time_cases(self):
        MNIST_10k = mnist_70k.head(10000).to_numpy()
        kmed1 = KMedoids(n_medoids = 5, algorithm = "BanditPAM", verbosity = 0)
        kmed1.fit(MNIST_10k, "L2")

        MNIST_schedule = [20, 40, 70]
        kmed2 = KMedoids(n_medoids = 5, algorithm = "BanditPAM", verbosity = 0)
        for num in MNIST_schedule:
            MNIST_test = mnist_70k.head(num * 1000).to_numpy()
            kmed2.fit(MNIST_test)
            self.assertTrue(kmed2.steps < ((num/10) ** 1.2) * kmed1.steps)

        SCRNA_10k = scrna.head(10000).to_numpy()
        kmed1.fit(SCRNA_10k, 'L1')

        scrna_schedule = [20, 30, 40]
        for num in scrna_schedule:
            scrna_test = scrna.head(num * 1000).to_numpy()
            kmed2.fit(MNIST_test)
            self.assertTrue(kmed2.steps < ((num/10) ** 1.2) * kmed1.steps)

if __name__ == '__main__':
    unittest.main()
