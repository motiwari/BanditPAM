import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from banditpam import KMedoids

# Load the 1000-point subset of MNIST and calculate its t-SNE embeddings for visualization:
X = pd.read_csv("data/MNIST-1k.csv", sep=" ", header=None).to_numpy()
X_tsne = TSNE(n_components=2).fit_transform(X)

# Fit the data with BanditPAM:
kmed = KMedoids(n_medoids=10, algorithm="BanditPAM")
kmed.fit(X, "L2", "mnist_log")

# Visualize the data and the medoids via t-SNE:
for p_idx, point in enumerate(X):
    if p_idx in map(int, kmed.medoids):
        plt.scatter(X_tsne[p_idx, 0], X_tsne[p_idx, 1], color="red", s=40)
    else:
        plt.scatter(X_tsne[p_idx, 0], X_tsne[p_idx, 1], color="blue", s=5)


plt.show()
