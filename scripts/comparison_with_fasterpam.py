from sklearn.datasets import fetch_openml
from collections import defaultdict
import time
import numpy as np
import banditpam
from collections import defaultdict

from sklearn.metrics.pairwise import euclidean_distances
from kmedoids import fasterpam

# Possible discrepancies: 
# - number of threads in BanditPAM vs. rust implementation
# - precomputation of entire distance matrix
# - Why is there a variable XX in the Google colab?
# - Euclidean_distances may be faster than the way BanditPAM computes it

def benchmark(data, f, n=1):
  data_ = defaultdict(list)
  for i in range(n):
    print("Restart", i)
    for k, v in f(data, i).items():
      data_[k].append(v)
  for k, v in data_.items():
    v = np.array(v)
    min, avg = v.min(), v.mean()
    ste = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.
    print("{:16s} min={:-10.2f} mean={:-10.2f} ±{:-.2f}".format(k, min, avg, ste))

def run_fasterpam(data, seed):
  start = time.time()
  diss = euclidean_distances(data) # compute distance matrix every time, to be fair to BanditPAM.
  mid = time.time()
  r = fasterpam(diss, 100, random_state=seed, n_cpu=1)
  end = time.time()
  meds, lbl = data[r.medoids], r.labels
  verified_loss = np.sqrt(((data - meds[lbl])**2).sum(axis=1)).sum()
  return {"time (ms)": (end-start)*1000, "verified loss": verified_loss, "reported loss": r.loss, "opt. time (ms)": (end-mid)*1000, "mat. time (ms)": (mid-start)*1000 }  

def run_bandit(data, seed):
  banditpam.set_num_threads(1) # c.f., https://github.com/ThrunGroup/BanditPAM/issues/176#issuecomment-1015920113
  km = banditpam.KMedoids(100)
  km.seed = seed
  start = time.time()
  km.fit(data, "L2")
  end = time.time()
  meds, lbl = data[km.medoids], km.labels
  #del km # try to free memory
  verified_loss = numpy.sqrt(((data - meds[lbl])**2).sum(axis=1)).sum()
  return {"time (ms)": (end-start)*1000, "verified loss": verified_loss, "reported loss": km.average_loss }  


if __name__ == "__main__":
    X, _ = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X[:15000] # at 20k, colab will timeout for BanditPAM
    print(X.shape, type(X))

    benchmark(X, run_fasterpam)
    benchmark(X, run_bandit)