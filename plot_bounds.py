import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

if __name__ == "__main__":
    lines = []
    with open("build/bounds.txt", 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 3):
        ucbs = [float(num) for num in [num for num in lines[i].split(' ')if not num.strip() == ""]]
        estimates = [float(num) for num in [num for num in lines[i + 1].split(' ')if not num.strip() == ""]]
        lcbs = [float(num) for num in [num for num in lines[i + 2].split(' ')if not num.strip() == ""]]
        plt.scatter(range(len(lcbs)), estimates, c = 'blue', marker = '.')
        plt.scatter(range(len(lcbs)), ucbs, c = 'green')
        plt.scatter(range(len(lcbs)), lcbs, c = 'red')
        x = []
        y = []
        for i in range(len(ucbs)):
            if (ucbs[i] - lcbs[i])**2 < .0001:
                x.append(i)
                y.append(estimates[i])
        plt.scatter(x, y, c = "cyan", marker='s')

        plt.show()
        for num in ucbs:
            print(num)
    #plt.scatter(data[:, 0], data[:, 1], c = assignments)
    #plt.show()
