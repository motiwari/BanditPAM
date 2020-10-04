'''
A number of convenience functions used by the different algorithms.
Also includes some constants.

There are 5 functions that call d and therefore require the an explicit metric:
- cost_fn
- cost_fn_difference
- cost_fn_difference_FP1
- get_best_distances
- estimate_sigma
- medoid_swap
'''

import os
import sys
import numpy as np
import mnist
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pickle

from zss import simple_distance, Node
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances

DECIMAL_DIGITS = 5
SIGMA_DIVISOR = 1

def get_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help = 'print debugging output', action = 'count', default = 0)
    parser.add_argument('-k', '--num_medoids', help = 'Number of medoids', type = int, default = 3)
    parser.add_argument('-N', '--sample_size', help = 'Sampling size of dataset', type = int, default = 700)
    parser.add_argument('-s', '--seed', help = 'Random seed', type = int, default = 42)
    parser.add_argument('-d', '--dataset', help = 'Dataset to use', type = str, default = 'MNIST')
    parser.add_argument('-c', '--cache_computed', help = 'Cache computed', default = None)
    parser.add_argument('-m', '--metric', help = 'Metric to use (L1 or L2)', type = str)
    parser.add_argument('-f', '--force', help = 'Recompute Experiments', action = 'store_true')
    parser.add_argument('-p', '--fast_pam1', help = 'Use FastPAM1 optimization', action = 'store_true')
    parser.add_argument('-r', '--fast_pam2', help = 'Use FastPAM2 optimization', action = 'store_true')
    parser.add_argument('-w', '--warm_start_medoids', help = 'Initial medoids to start with', type = str, default = '')
    parser.add_argument('-B', '--build_ao_swap', help = 'Build or Swap, B = just build, S = just swap, BS = both', type = str, default = 'BS')
    parser.add_argument('-e', '--exp_config', help = 'Experiment configuration file to use', required = False)
    args = parser.parse_args(arguments)
    return args

def load_data(args):
    '''
    Load the different datasets, as a numpy matrix if possible. In the case of
    HOC4 and HOC18, load the datasets as a list of trees.
    '''
    if args.dataset == 'MNIST':
        N = 70000
        m = 28
        sigma = 0.7
        train_images = mnist.train_images()
        train_labels = mnist.train_labels()

        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        total_images = np.append(train_images, test_images, axis = 0)
        total_labels = np.append(train_labels, test_labels, axis = 0)

        assert((total_images == np.vstack((train_images, test_images))).all())
        assert((total_labels == np.hstack((train_labels, test_labels))).all()) # NOTE: hstack since 1-D
        assert(total_images.shape == (N, m, m))
        assert(total_labels.shape == (N,))

        # Normalizing images
        return total_images.reshape(N, m * m) / 255, total_labels, sigma
    elif args.dataset == "SCRNA":
        file = 'person1/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/NUMPY_OUT/np_data.npy'
        data_ = np.load(file)
        sigma = 25
        return data_, None, sigma
    elif args.dataset == "SCRNAPCA":
        file = 'person1/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/analysis_csv/pca/projection.csv'
        df = pd.read_csv(file, sep=',', index_col = 0)
        np_arr = df.to_numpy()
        sigma = 0.01
        return np_arr, None, sigma
    elif args.dataset == 'HOC4':
        dir_ = 'hoc_data/hoc4/trees/'
        tree_files = [dir_ + tree for tree in os.listdir(dir_) if tree != ".DS_Store"]
        trees = []
        for tree_f in sorted(tree_files):
            with open(tree_f, 'rb') as fin:
                tree = pickle.load(fin)
                trees.append(tree)

        if args.verbose >= 1:
            print("NUM TREES:", len(trees))

        return trees, None, 0.0
    elif args.dataset == 'HOC18':
        dir_ = 'hoc_data/hoc18/trees/'
        tree_files = [dir_ + tree for tree in os.listdir(dir_) if tree != ".DS_Store"]
        trees = []
        for tree_f in tree_files:
            with open(tree_f, 'rb') as fin:
                tree = pickle.load(fin)
                trees.append(tree)

        if args.verbose >= 1:
            print("NUM TREES:", len(trees))

        return trees, None, 0.0
    elif args.dataset == 'GAUSSIAN':
        dataset = create_gaussians(args.sample_size, ratio = 0.6, seed = args.seed, visualize = False)
        return dataset
    else:
        raise Exception("Didn't specify a valid dataset")



def visualize_medoids(dataset, medoids, visualization = 'tsne'):
    '''
    Helper function to visualize the given medoids of a dataset using t-SNE
    '''

    if visualization == 'tsne':
        X_embedded = TSNE(n_components=2).fit_transform(dataset)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='b')
        plt.scatter(X_embedded[medoids, 0], X_embedded[medoids, 1], c='r')
        plt.show()
    else:
        raise Exception('Bad Visualization Arg')

def create_gaussians(N, ratio = 0.6, seed = 42, visualize = True):
    '''
    Create some 2-D Gaussian toy data.
    '''

    np.random.seed(seed)
    cluster1_size = int(N * ratio)
    cluster2_size = N - cluster1_size

    cov1 = np.array([[1, 0], [0, 1]])
    cov2 = np.array([[1, 0], [0, 1]])

    mu1 = np.array([-10, -10])
    mu2 = np.array([10, 10])

    cluster1 = np.random.multivariate_normal(mu1, cov1, cluster1_size)
    cluster2 = np.random.multivariate_normal(mu2, cov2, cluster2_size)

    if visualize:
        plt.scatter(cluster1[:, 0], cluster1[:, 1], c='r')
        plt.scatter(cluster2[:, 0], cluster2[:, 1], c='b')
        plt.show()

    return np.vstack((cluster1, cluster2))



if __name__ == "__main__":
    create_gaussians(1000, 0.5, 42)

    ####### Use the code below to visualize the some medoids with t-SNE
    # args = get_args(sys.argv[1:])
    # total_images, total_labels, sigma = load_data(args)
    # np.random.seed(args.seed)
    # imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    # visualize_medoids(imgs, [891, 392])
