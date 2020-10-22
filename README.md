# BanditPAM: Almost Linear-Time *k*-Medoids Clustering

This repo contains a high-performance implementation of BanditPAM from https://arxiv.org/abs/2006.06856. The code can be called directly from Python or C++.

If you use this software, please cite:

Mo Tiwari, Martin Jinye Zhang, James Mayclin, Sebastian Thrun, Chris Piech, Ilan Shomorony. "Bandit-PAM: Almost Linear Time *k*-medoids Clustering via Multi-Armed Bandits" Neural Information Processing Systems (NeurIPS) 2020.


```python
@inproceedings{BanditPAM,
  title={Bandit-PAM: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits},
  author={Tiwari, Mo and Zhang, Martin J and Mayclin, James and Thrun, Sebastian and Piech, Chris and Shomorony, Ilan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={368--374}, #TODO: Fix this
  year={2020}
}
```

# Python Quickstart

## Install the repo and its dependencies:


```python
/BanditPAM/: pip install -r requirements.txt
/BanditPAM/: sudo pip install .
```

### Example 1: Synthetic data from a Gaussian Mixture Model


```python
from BanditPAM import KMedoids
import numpy as np
import matplotlib.pyplot as plt

# Generate data from a Gaussian Mixture Model with the given means:
np.random.seed(0)
n_per_cluster = 40
means = np.array([[0,0], [-5,5], [5,5]])
X = np.vstack([np.random.randn(n_per_cluster, 2) + mu for mu in means])

# Fit the data with BanditPAM:
kmed = KMedoids(n_medoids = 3, algorithm = "BanditPAM")
# Writes results to gmm_log
kmed.fit(X, 'L2', 3, "gmm_log")

# Visualize the data and the medoids:
for p_idx, point in enumerate(X):
    if p_idx in map(int, kmed.final_medoids):
        plt.scatter(X[p_idx, 0], X[p_idx, 1], color='red', s = 40)
    else:
        plt.scatter(X[p_idx, 0], X[p_idx, 1], color='blue', s = 10)
plt.show()
```


![png](README_files/README_7_0.png)


### Example 2: MNIST and its medoids visualized via t-SNE


```python
# Start in the repository root directory, i.e. '/BanditPAM/'.
from BanditPAM import KMedoids
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the 1000-point subset of MNIST and calculate its t-SNE embeddings for visualization:
X = pd.read_csv('data/MNIST-1k.csv', sep=' ', header=None).to_numpy()
X_tsne = TSNE(n_components = 2).fit_transform(X)

# Fit the data with BanditPAM:
kmed = KMedoids(n_medoids = 10, algorithm = "BanditPAM")
kmed.fit(X, 'L2', 10, "mnist_log")

# Visualize the data and the medoids via t-SNE:
for p_idx, point in enumerate(X):
    if p_idx in map(int, kmed.final_medoids):
        plt.scatter(X_tsne[p_idx, 0], X_tsne[p_idx, 1], color='red', s = 40)
    else:
        plt.scatter(X_tsne[p_idx, 0], X_tsne[p_idx, 1], color='blue', s = 5)
plt.show()
```
The corresponding logfile for this run, `mnist_log`, will contain the run's results
and additional statistics in a format that can be easily read into json.

## Building the C++ executable from source

Please note that it is not necessary to build the C++ executable from source to use the Python code above. However, if you would like to use the C++ executable directly, follow the instructions below.

### Option 1: Building with Docker

We highly recommend building using Docker. One can download and install Docker by following instructions at the [Docker install page](https://docs.docker.com/get-docker/). Once you have Docker installed and the Docker Daemon is running, run the following commands:

```
/BanditPAM$ chmod +x env_setup.sh
/BanditPAM$ ./env_setup.sh
/BanditPAM$ ./run_docker.sh
```

which will start a Docker instance with the necessary dependencies. Then:

```
/BanditPAM$ mkdir build && cd build
/BanditPAM/build$ cmake .. && make
```

This will create an executable named `BanditPAM` in `BanditPAM/build/src`.

### Option 2: Installing Requirements and Building Directly
Building this repository requires three external requirements:
* Cmake >= 3.17, https://cmake.org/download/
* Armadillo >= 9.7, http://arma.sourceforge.net/download.html
* OpenMP >= 2.5, https://www.openmp.org/resources/openmp-compilers-tools/ (OpenMP is supported by default on most Linux platforms, and can be downloaded through homebrew on MacOS. For instructions on installing homebrew, see https://brew.sh/.)
* CARMA >= 0.3.0, https://github.com/RUrlus/carma

If installing these requirements from source, one can generally use the following procedure to install each requirement from the library's root folder (with CARMA used as an example here):
```
/carma$ mkdir build && cd build
/carma/build$ cmake .. && make && make install
```

Ensure all the requirements above are installed and then run:

```
/BanditPAM$ mkdir build && cd build
/BanditPAM/build$ cmake .. && make
```

This will create an executable named `BanditPAM` in `BanditPAM/build/src`.

## Usage

Once the executable has been built, it can be invoked with:
```
/BanditPAM/build/src/BanditPAM -f [path/to/input.csv] -k [number of clusters] -v [verbosity level]
```

* `-f` is mandatory and specifies the path to the dataset
* `-k` is mandatory and specifies the number of clusters with which to fit the data
* `-v` is optional and specifies the verbosity level.

For example, if you ran `./env_setup.sh` and downloaded the MNIST dataset, you could run:

```
/BanditPAM/build/src/BanditPAM -f ../data/MNIST-1k.csv -k 10 -v 1
```

The expected output in the command line will be:
```
Medoids: 694,168,306,714,324,959,527,251,800,737
```
A file called `KMedoidsLogfile` with detailed logs during the process will also
be present.

## Testing

To run the full suite of tests, run in the root directory:

```
/BanditPAM$ python -m unittest discover -s tests
```

Alternatively, to run a "smaller" set of tests, from the main repo folder run `python tests/test_commit.py` or `python tests/test_push.py` to run a set of longer, more intensive tests.
