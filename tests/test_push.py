import unittest
from BanditPAM import KMedoids
import pandas as pd
import numpy as np

small_mnist = pd.read_csv('./data/mnist.csv', sep=' ', header=None).to_numpy()

mnist_70k = pd.read_csv('./data/MNIST-70k.csv', sep=' ', header=None)

scrna = pd.read_csv('./data/scrna_reformat.csv.gz', header=None)

def onFly(k, data):
    kmed_bpam = KMedoids(k = k, algorithm = "BanditPAM")
    kmed_naive = KMedoids(k = k, algorithm = "naive")
    kmed_bpam.fit(data)
    kmed_naive.fit(data)
    # TODO: do we need to check build?
    if (kmed_bpam.final_medoids == kmed_naive.final_medoids):
        return 1
    else:
        return 0

class PythonTests(unittest.TestCase):
    def large_on_fly_test_cases(self):
        count = 0
        k_schedule = [4, 6, 8, 10] * 25
        size_schedule = [1000, 2000, 3000, 4000, 5000] * 20
        for i in range(50): #arbitrary heuristic
            data = mnist_70k.sample(n = size_schedule[i])
            count += onFly(k = k_schedule[i], data = data)

        for i in range(50, 100):
            data = scrna.sample(n = size_schedule[i])
            count += onFly(k = k_schedule[i], data = data)

        self.assertTrue(count >= 95)

    def time_test_cases(self):
        MNIST_10k = mnist_70k.head(10000).to_numpy()
        kmed1 = KMedoids(n_medoids = 5, algorithm = "BanditPAM", verbosity = 0)
        kmed1.fit(MNIST_10k)

        MNIST_schedule = [20, 40, 70]
        kmed2 = KMedoids(n_medoids = 5, algorithm = "BanditPAM", verbosity = 0)
        for num in MNIST_schedule:
            MNIST_test = mnist_70k.head(num * 1000).to_numpy()
            kmed2.fit(MNIST_test)
            self.assertTrue(kmed2.steps < ((num/10) ** 1.2) * kmed1.steps)

        SCRNA_10k = scrna.head(10000).to_numpy()
        kmed1.fit(SCRNA_10k)

        scrna_schedule = [20, 30, 40]
        for num in scrna_schedule:
            scrna_test = scrna.head(num * 1000).to_numpy()
            kmed2.fit(MNIST_test)
            self.assertTrue(kmed2.steps < ((num/10) ** 1.2) * kmed1.steps)

if __name__ == '__main__':
    unittest.main()
