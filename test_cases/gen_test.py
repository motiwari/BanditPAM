import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

k = 4
d = 5
N = 3000
buf_distance = 2

def assigned(attempt, medoids, m_i):
    best_distance = np.inf
    best_ind = 0
    for i in range(d):
        cost = np.linalg.norm(attempt - medoids[i, :])
        if cost < best_distance:
            best_distance = cost
            best_ind = i
    return m_i == best_ind

def gen_medoids():
    medoid = np.random.normal(scale = 20, size = (k, d))
    return medoid

def gen_adjacent(medoids):
    data = np.zeros(shape = (N, d))
    assignments = np.zeros(shape = (N,), dtype=int)
    i = 0
    for i in range(N):
        print("generating point")
        assignment = np.random.randint(k)
        cur_med = medoids[assignment, :]
        attempt = np.random.normal(loc = cur_med, size = (1, d))
        # too close to center to too close to other
        distance = np.linalg.norm(attempt - cur_med)
        while distance < buf_distance:
            attempt = np.random.normal(loc = cur_med, size = (1, d))
            distance = np.linalg.norm(attempt - cur_med)

        data[i, :] = attempt
        assignments[i] = assignment

    # write actual medoids to random points
    medoid_indices = np.zeros(shape = (k,))
    for i in range(k):
        index = i * (N// k + 1) + np.random.randint(N// k + 1)

        data[index, :] = medoids[i, :]
        assignments[index] = i 
        medoid_indices[i] = index
    return data, assignments, medoid_indices

def calc_loss(data, assignments, medoids):
    total = 0
    for i in range(N):
        point = data[i, :]
        print(assignments[i])
        medoid = medoids[assignments[i], :]
        total += np.linalg.norm(point - medoid)
    return total

if __name__ == "__main__":
    medoids = gen_medoids()
    data, assignments, medoid_indices = gen_adjacent(medoids)
    print(assignments)
    if (d == 2):
        plt.scatter(data[:, 0], data[:, 1], c = assignments)
        plt.show()
    print("loss is {}".format(calc_loss(data, assignments, medoids)))
    kmedoids = KMedoids(n_clusters=k, random_state=0).fit(data)
    print(kmedoids.cluster_centers_)
    print("here are medoids")
    print(medoids)
    if (input("Save dataset? (y)es or (n)o:") == "y"):
        prefix = input("save prefix:")
        np.savetxt(f"{prefix}.csv", data, delimiter=",")
        np.savetxt(f"{prefix}_medoids.csv", medoids, delimiter=",")
        np.savetxt(f"{prefix}_indices.csv", medoid_indices, fmt = "%i", delimiter=",")
        np.savetxt(f"{prefix}_assignments.csv", assignments, fmt = "%i", delimiter=",")
