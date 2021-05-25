import unittest
from BanditPAM import KMedoids
import pandas as pd
import numpy as np

def onFly(k, data, loss):
    kmed_bpam = KMedoids(
        n_medoids = k,
        algorithm = "BanditPAM",
    )
    kmed_naive = KMedoids(
        n_medoids = k,
        algorithm = "naive",
    )
    kmed_bpam.fit(data, loss)
    kmed_naive.fit(data, loss)

    if (kmed_bpam.medoids.tolist() == kmed_naive.medoids.tolist()) and \
       (kmed_bpam.build_medoids.tolist() == kmed_naive.build_medoids.tolist()):
        return 1
    else:
        return 0

class PythonTests(unittest.TestCase):
    small_mnist = pd.read_csv('./data/MNIST.csv', header=None).to_numpy()

    mnist_70k = pd.read_csv('./data/MNIST-70k.csv', sep=' ', header=None)

    scrna = pd.read_csv('./data/scrna_reformat.csv.gz', header=None)

    def test_small_on_fly_mnist(self):
        '''
        Test 10 on-the-fly generated samples of 100 datapoints from mnist-70k dataset
        '''
        count = 0
        k_schedule = [4, 6, 8, 10] * 3
        for i in range(10):
            data = self.mnist_70k.sample(n = 100).to_numpy()
            count += onFly(k = k_schedule[i], data = data, loss = "L2")
        self.assertTrue(count >= 9)

    def test_small_on_fly_scrna(self):
        '''
        Test 10 on-the-fly generated samples of 100 datapoints from scrna dataset
        '''
        count = 0
        k_schedule = [4, 6, 8, 10] * 3
        for i in range(10):
            data = self.scrna.sample(n = 100).to_numpy()
            count += onFly(k = k_schedule[i], data = data, loss = "L1")
        self.assertTrue(count >= 9)

    def test_small_cases_mnist(self):
        '''
        Test BanditPAM algorithm at 5 and 10 medoids on mnist dataset with known medoids
        '''
        kmed_5 = KMedoids(
            n_medoids = 5,
            algorithm = "BanditPAM",
            max_iter = 1000,
            verbosity = 0,
            logFilename = "KMedoidsLogfile",
        )
        kmed_5.fit(self.small_mnist, "L2")

        self.assertEqual(kmed_5.build_medoids.tolist(), np.array([16, 32, 70, 87, 24]).tolist())
        self.assertEqual(kmed_5.medoids.tolist(), np.array([30, 99, 70, 23, 49]).tolist())

        kmed_10 = KMedoids(
            n_medoids = 10,
            algorithm = "BanditPAM",
            max_iter = 1000,
            verbosity = 0,
            logFilename = "KMedoidsLogfile",
        )
        kmed_10.fit(self.small_mnist, "L2")

        self.assertEqual(kmed_10.build_medoids.tolist(), np.array([16, 32, 70, 87, 24, 90, 49, 99, 82, 94]).tolist())
        self.assertEqual(kmed_10.medoids.tolist(), np.array([16, 63, 70, 25, 31, 90, 49, 99, 82, 94]).tolist())

    def test_small_cases_scrna(self):
        '''
        Test BanditPAM algorithm at 5 and 10 medoids on scrna dataset with known medoids
        '''
        kmed_5 = KMedoids(
            n_medoids = 5,
            algorithm = "BanditPAM",
            max_iter = 1000,
            verbosity = 0,
            logFilename = "KMedoidsLogfile",
        )
        kmed_5.fit(self.scrna.head(1000).to_numpy(), "L1")

        self.assertEqual(kmed_5.build_medoids.tolist(), np.array([377, 267, 276, 762, 394]).tolist())
        self.assertEqual(kmed_5.medoids.tolist(), np.array([377, 267, 276, 762, 394]).tolist())

        kmed_10 = KMedoids(
            n_medoids = 10,
            algorithm = "BanditPAM",
            max_iter = 1000,
            verbosity = 0,
            logFilename = "KMedoidsLogfile",
        )
        kmed_10.fit(self.scrna.head(1000).to_numpy(), "L1")

        self.assertEqual(kmed_10.build_medoids.tolist(), np.array([377, 267, 276, 762, 394, 311, 663, 802, 422, 20]).tolist())
        self.assertEqual(kmed_10.medoids.tolist(), np.array([377, 267, 276, 762, 394, 311, 663, 802, 422, 20]).tolist())

    def test_edge_cases(self):
        '''
        Test BanditPAM algorithm on edge cases
        '''
        kmed = KMedoids()

        # initialized to empty
        self.assertEqual([], kmed.medoids.tolist())
        self.assertEqual([], kmed.build_medoids.tolist())

        # error on trying to fit on empty
        self.assertRaises(RuntimeError, kmed.fit(np.array([])), "L2")

if __name__ == '__main__':
    unittest.main()
