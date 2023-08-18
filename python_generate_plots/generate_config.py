'''
Convenience code to automatically generate a list of experiments to run.
Default output is to auto_exp_config.py.
'''

import itertools

def write_exp(algo, k, N, seed, dataset, metric):
    '''
    Takes the experiment variables and outputs a string description
    to go into a config file.
    '''
    return "\t['" + algo + "', 'BS', 0, " + str(k) + ", " + str(N) + \
        ", " + str(seed) + ", '" + dataset + "', '" + metric + "', ''],\n"

def main():
    # TODO(@Adarsh321123): change comments throughout
    # TODO(@Adarsh321123): remove unnecessary things
    # Possible algos are ['ucb', 'naive_v1', 'em_style', 'csh', and 'clarans']
    # algos = ['naive_v1']
    # seeds = range(10)

    ######## HOC4, Tree edit distance (precomputed), k = 2 and k = 3
    # dataset = 'HOC4'
    # metric = 'PRECOMP'
    # Ns = [1000, 2000, 3000, 3360]
    # ks = [2]
    # seeds = range(10)
    # algos = ['ucb']

    #### for MNIST L2, k = 3
    # dataset = 'MNIST'
    # metric = 'L2'
    # # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000]
    # # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    # Ns = [5000, 6000, 7500, 10000]
    # ks = [3]
    # # seeds = range(30)
    # seeds = range(10)
    # algos = ['bfp']
    # # algos = ['bp']

    #### for MNIST L2, k = 5
    # dataset = 'MNIST'
    # metric = 'L2'
    # # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000]
    # # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    # Ns = [5000, 6000, 7500, 10000]
    # ks = [5]
    # # seeds = range(30)
    # seeds = range(10)
    # algos = ['bfp']
    # # algos = ['bp']

    ######## MNIST, Cosine distance, k = 3
    # dataset = 'MNIST'
    # metric = 'cosine'
    # # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000]
    # # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    # Ns = [5000, 6000, 7500, 10000]
    # ks = [3]
    # # seeds = range(30)
    # seeds = range(10)
    # algos = ['bfp']
    # # algos = ['bp']

    ######## SCRNA, L1 distance, k = 3
    dataset = 'SCRNA'
    metric = 'L1'
    # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000]
    Ns = [5000, 6000, 7500, 10000]
    ks = [3]
    # seeds = range(30)
    seeds = range(10)
    algos = ['bfp']

    ######## SCRNAPCA, L2 distance, k = 5 and k = 10
    # dataset = 'SCRNAPCA'
    # Ns = [3000, 10000, 20000, 30000, 40000]
    # ks = [5, 10]
    # metric = 'L2'

    ######## CIFAR, L1 distance, k = 2
    # dataset = 'CIFAR'
    # metric = 'L1'
    # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000]
    # ks = [2]
    # seeds = range(30)
    # algos = ['bfp']

    # ######## For loss plots
    dataset = 'MNIST'
    metric = 'L2'
    # algos = ['naive_v1', 'bfp', 'fp']
    algos = ['naive_v1']
    seeds = range(10)
    # Ns = [500, 1000, 1500, 2000, 2500, 3000]
    # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    Ns = [5000, 7500, 10000]
    # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000]
    ks = [5]

    with open('auto_exp_config.py', 'w+') as fout:
        fout.write("experiments = [\n")
        for k in ks:
            for seed in seeds:
                for N in Ns:
                    for algo in algos:
                        # Adding 42 to seed for comparison with earlier experiments
                        exp = write_exp(algo, k, N, 42 + seed, dataset, metric)
                        if exp is not None:
                            fout.write(exp)
        fout.write("]")

if __name__ == "__main__":
    main()
