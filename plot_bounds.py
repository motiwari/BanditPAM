'''
Debugging function that can be used for visualizing the update of
confidence bounds and estimates on successive iterations of the swap
or build steps.

Expected format of bounds.txt is triplets of lines where the first line
is a space separated list of lcbs, second line is a space separated list
of estimates, and third line is a space separated list of ucbs.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

DOUBLE_COMPARISON_BOUND = 0.001

def parse_raw(line):
    tokens = [num for num in lines[i].split(' ')if num.strip() != ""]
    numbers = [float(num) for num in tokens]
    return numbers

def main():
    # read in all lines
    lines = []
    with open("build/bounds.txt", 'r') as f:
        lines = f.readlines()
    
    # process all lines
    for i in range(0, len(lines), 3):
        # parse in lines
        ucbs = parse_raw(lines[i])
        estimates = parse_raw(lines[i + 1])
        lcbs = parse_raw(lines[i + 2])

        # plot bounds and estimates
        N = len(estimates)
        plt.scatter(range(N), estimates, c = 'blue', marker = '.')
        plt.scatter(range(N), ucbs, c = 'green')
        plt.scatter(range(N), lcbs, c = 'red')
        
        # plot converged points
        x = []
        y = []
        for i in range(len(ucbs)):
            if (ucbs[i] - lcbs[i]) < DOUBLE_COMPARISON_BOUND:
                x.append(i)
                y.append(estimates[i])
        plt.scatter(x, y, c = "cyan", marker='s')

        plt.show()

if __name__ == "__main__":
    main()
