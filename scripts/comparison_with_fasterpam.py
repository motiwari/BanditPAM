# Primarily authored by Erich Schubert (@kno10)

from sklearn.datasets import fetch_openml
from collections import defaultdict
import time
import numpy as np
import banditpam

from sklearn.metrics.pairwise import euclidean_distances
from kmedoids import fasterpam, fastpam1

# - Why is there a variable XX in the Google colab?


def benchmark(data, f, n=1):
    data_ = defaultdict(list)
    for i in range(n):
        print("Restart", i)
        for k, v in f(data, i).items():
            data_[k].append(v)

    for k, v in data_.items():
        v = np.array(v)
        min, avg = v.min(), v.mean()
        ste = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        print("{:16s} min={:-10.2f} mean={:-10.2f} ±{:-.2f}".format(k, min, avg, ste))


def run_fasterpam(data, seed):
    diss = euclidean_distances(data)
    start = time.time()
    r = fasterpam(diss, 5, random_state=seed)
    end = time.time()
    print("FasterPAM took ", r.n_iter, " iterations")
    meds, lbl = data[r.medoids], r.labels
    verified_loss = np.sqrt(((data - meds[lbl]) ** 2).sum(axis=1)).sum()
    return {
        "time (ms)": (end - start),
        "verified loss": verified_loss,
        "reported loss": r.loss,
    }


def run_bandit(data, seed):
    diss = euclidean_distances(data)
    km = banditpam.KMedoids(5, parallelize=True)
    print(km.algorithm)
    km.seed = seed
    start = time.time()
    km.fit(data, "L2")
    end = time.time()
    meds, lbl = data[km.medoids], km.labels
    verified_loss = np.sqrt(((data - meds[lbl]) ** 2).sum(axis=1)).sum()
    return {
        "time (ms)": (end - start),
        "verified loss": verified_loss,
        "reported loss": km.average_loss,
    }

def run_old_bandit(data, seed):
    diss = euclidean_distances(data)
    km = banditpam.KMedoids(5, parallelize=True, algorithm="BanditPAM_v3")
    print(km.algorithm)
    km.seed = seed
    start = time.time()
    km.fit(data, "L2")
    end = time.time()
    meds, lbl = data[km.medoids], km.labels
    verified_loss = np.sqrt(((data - meds[lbl]) ** 2).sum(axis=1)).sum()
    return {
        "time (ms)": (end - start),
        "verified loss": verified_loss,
        "reported loss": km.average_loss,
    }


if __name__ == "__main__":
    X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X[:20000]  # at 20k, colab will timeout for BanditPAM
    print(X.shape, type(X))

    # print("Benchmarking FasterPAM")
    # benchmark(X, run_fasterpam)

    print("\n\nBenchmarking BanditPAM")
    benchmark(X, run_bandit)

    print("\n\nBenchmarking Old BanditPAM")
    benchmark(X, run_old_bandit)


