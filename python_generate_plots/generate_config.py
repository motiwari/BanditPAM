'''
Convenience code to automatically generate a list of experiments to run.
Default output is to auto_exp_config.py.
'''

def write_exp(algo, k, N, seed, dataset, metric):
    '''
    Takes the experiment variables and outputs a string description
    to go into a config file.
    '''
    return "\t['" + algo + "', " + str(k) + ", " + str(N) + \
        ", " + str(seed) + ", '" + dataset + "', '" + metric + "'],\n"

def main():
    # Possible algos are ['bfp', 'fp', 'naive_v1']
    algos = ['bfp']
    seeds = range(30)

    ######## Figure 1 (a): loss plots
    # dataset = 'MNIST'
    # metric = 'L2'
    # Ns = [5000, 7500, 10000]
    # ks = [5]
    # algos = ['bfp', 'fp', 'naive_v1']
    # seeds = range(10)

    ######## Figure 1 (b): CIFAR, L1, k = 2
    # dataset = 'CIFAR'
    # metric = 'L1'
    # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    # ks = [2]

    ######## Figure 2 (a): MNIST, L2, k = 3
    dataset = 'MNIST'
    metric = 'L2'
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    ks = [3]

    ######## Figure 2 (b): MNIST, L2, k = 5
    # dataset = 'MNIST'
    # metric = 'L2'
    # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    # ks = [5]

    ######## Figure 3 (a): MNIST, cosine, k = 3
    # dataset = 'MNIST'
    # metric = 'cosine'
    # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    # ks = [3]

    ######## Figure 3 (b): SCRNA, L1, k = 3
    # dataset = 'SCRNA'
    # metric = 'L1'
    # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    # ks = [3]

    with open('auto_exp_config.py', 'w+') as fout:
        fout.write("experiments = [\n")
        for k in ks:
            for seed in seeds:
                for N in Ns:
                    for algo in algos:
                        exp = write_exp(algo, k, N, 42 + seed, dataset, metric)
                        if exp is not None:
                            fout.write(exp)
        fout.write("]")

if __name__ == "__main__":
    main()
