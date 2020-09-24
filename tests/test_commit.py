import unittest
from banditPAM import KMedoids
import pandas as pd
import numpy as np

small_mnist = pd.read_csv('./data/mnist.csv', header=None).to_numpy()

mnist_70k = pd.read_csv('./data/MNIST-70k.csv', sep=' ', header=None)

scrna = pd.read_csv('./data/scrna_reformat.csv.gz', header=None)

def onFly(k, data):
    kmed_bpam = KMedoids(k = k, algorithm = "BanditPAM")
    kmed_naive = KMedoids(k = k, algorithm = "naive")
    kmed_bpam.fit(data)
    kmed_naive.fit(data)
    # TODO: do we need to check build
    if (kmed_bpam.final_medoids == kmed_naive.final_medoids):
        return 1
    else:
        return 0

class PythonTests(unittest.TestCase):
    def small_on_fly_test_cases(self):
        count = 0
        k_schedule = [4, 6, 8, 10] * 5
        for i in range(10): #arbitrary heuristic
            data = mnist_70k.sample(n = 1000).to_numpy()
            count += onFly(k = k_schedule[i], data = data)

        for i in range(10): #arbitrary heuristic
            data = scrna.sample(n = 1000).to_numpy()
            count += onFly(k = k_schedule[i], data = data)
        self.assertTrue(count >= 19)

    def small_test_cases(self):
        kmed_5 = KMedoids(
            n_medoids = 5,
            algorithm = "BanditPAM",
            max_iter = 1000,
            verbosity = 0,
            logFilename = "KMedoidsLogfile"
        )
        kmed_5.fit(small_mnist, "L2")
        # TODO FINISH
        self.assertEqual(kmed_5.build_medoids, np.array([16, 32, 70, 87, 24]))
        self.assertEqual(kmed_5.final_medoids, np.array([30, 99, 70, 23, 49]))

        kmed_5.fit(scrna.head(1000).to_numpy(), "L1")
        self.assertEqual(kmed_5.build_medoids, np.array([377, 267, 276, 762, 394]))
        self.assertEqual(kmed_5.final_medoids, np.array([377, 267, 276, 762, 394]))

        kmed_10 = KMedoids(
            n_medoids = 10,
            algorithm = "BanditPAM",
            max_iter = 1000,
            verbosity = 0,
            logFilename = "KMedoidsLogfile"
        )
        kmed_10.fit(small_mnist, "L2")
        # TODO FINISH
        self.assertEqual(kmed_10.build_medoids, np.array([16, 32, 70, 87, 24, 90, 49, 99, 82, 94]))
        self.assertEqual(kmed_10.final_medoids, np.array([16, 63, 70, 25, 31, 90, 49, 99, 82, 94]))

        kmed_10.fit(scrna.head(1000).to_numpy(), "L1")
        # TODO FINISH
        self.assertEqual(kmed_10.build_medoids, np.array([377, 267, 276, 762, 394, 311, 663, 802, 422, 20]))
        self.assertEqual(kmed_10.final_medoids, np.array([377, 267, 276, 762, 394, 311, 663, 802, 422, 20]))

    def edge_cases(self):
        kmed = KMedoids()

        # initialized to empty
        self.assertEqual([], kmed.final_medoids.tolist())
        self.assertEqual([], kmed.build_medoids.tolist())

        # error on trying to fit on empty
        self.assertRaises(RuntimeError, kmed.fit(np.array([])))

        # TODO: error if the number of medoids larger than num data points



if __name__ == '__main__':
    unittest.main()
