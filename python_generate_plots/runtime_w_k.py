'''
Code to automatically parse the profiles produced from running experiments.
In particular, plots the scaling of BanditPAM vs. N for various dataset sizes N.
Used to demonstrate O(NlogN) scaling.
'''

import sys
import os
import pstats
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from generate_config import write_exp
import constants


def plot_slice_sns(dcalls_array, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap, title = "Insert title", take_log = True):
    '''
    Plots the number of distance calls vs. N, for various algorithms, seeds,
    values of k, and weightings between build and swap.

    Requires the array of distance calls for the algo, for each k, N, and seed.
    '''
    assert fix_k_or_N == 'N' or fix_k_or_N == 'k', "Bad slice param"

    # Determine what we're fixing and what we're plotting the scaling against
    if fix_k_or_N == 'k':
        kNs = ks
        Nks = Ns
    elif fix_k_or_N == 'N':
        kNs = Ns
        Nks = ks

    for kN_idx, kN in enumerate(kNs):
        if take_log:
            np_data = np.log10(dcalls_array)
            Nks_plot = np.log10(Nks)
        else:
            np_data = dcalls_array
            Nks_plot = Nks

        sns.set()
        sns.set_style('white')

        fig, ax = plt.subplots(figsize = (7,7))

        # Make a dataframe with the relevant data, for plotting with seaborn
        d = {'N': Nks_plot}
        for seed_idx, seed in enumerate(seeds):
            d["seed_" + str(seed)] = np_data[:, kN_idx, seed_idx]
        df = pd.DataFrame(data = d)
        print(df)

        # Combine the different seeds into 1 column
        melt_df = df.melt('N', var_name='cols', value_name='vals')
        melt_df['N'] += np.random.randn(melt_df['N'].shape[0]) * 0.01 # Add jitter
        sns.scatterplot(x="N", y="vals", data = melt_df, ax = ax, alpha = 0.6)

        # Plot means and error bars
        bars = (1.96/(10**0.5)) * np.std(np_data[:, kN_idx, :], axis = 1) # Slice a specific k, get a 2D array
        means = np.mean(np_data[:, kN_idx, :], axis = 1)
        plt.errorbar(Nks_plot, means, yerr = bars, fmt = '+', capsize = 5, ecolor='black', elinewidth = 1.5, zorder = 100, mec='black', mew = 1.5, label="95% confidence interval")

        # Plot line of best fit
        sl, icpt, r_val, p_val, _ = sp.stats.linregress(Nks_plot, means)
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        plt.plot([x_min, x_max], [x_min * sl + icpt, x_max * sl + icpt], color='black', label='Linear fit, slope=%0.3f'%(sl))


        # Manually modify legend labels for prettiness
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles[::-1], labels[::-1], loc="upper left")

        # elif fix_k_or_N == 'N':
        #     raise Exception("Fixing N and plotting vs. k not yet supported")

        plt.xlabel("$\log_{10}(k)$")
        plt.ylabel("$\log_{10}$(runtime (s))")

        # Modify these lines based on dataset
        plt.title(title)
        plt.savefig('figures/' + title.replace("$", '') + '.pdf')

def get_swap_T(logfile):
    '''
    Get the number of swap steps performed in an experiment, from parsing the
    logfile
    '''

    with open(logfile, 'r') as fin:
        line = fin.readline()
        while line[:10] != 'Num Swaps:':
            line = fin.readline()

        T = int(line.split(':')[-1])
    return T


def get_runtime(timefile):
    '''
    Get the runtime of an experiment, from parsing the timefile
    '''

    with open(timefile, 'r') as fin:
        line = fin.readline()
        print(line)
        while line[:8] != 'Runtime:':
            print(line)
            line = fin.readline()

        runtime = float(line.split(':')[-1].strip())
    return runtime

def show_plots(fix_k_or_N, build_or_swap, Ns, ks, seeds, algo, dataset, metric, dir_, title = "Insert title"):
    '''
    A function which mines the number of distance calls for each experiment,
    from the dumped profiles. Creates a numpy array with the distance call
    counts.

    It does this by:
        - first, identifying the filenames where the experiment profiles and
            logfiles are stored (the logfile is used for the number of swap
            steps)
        - searching each profile (build and swap) for the number of distance
            calls
        - weighting the distance calls between the build step and swap step as
            necessary
    '''
    runtimes = np.zeros((len(ks), len(Ns), len(seeds)))
    log_prefix = 'profiles/p-'
    time_prefix = 'profiles/t-'

    # Gather data
    for N_idx, N in enumerate(Ns):
        for k_idx, k in enumerate(ks):
            for seed_idx, seed in enumerate(seeds):
                # Get the time
                time_fname = time_prefix + 'ucb-True-BS-v-0-k-' + str(k) + \
                    '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                # Get the number of swaps
                logfile = log_prefix + 'ucb-True-BS-v-0-k-' + str(k) + \
                    '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                rt = get_runtime(time_fname)
                T = get_swap_T(logfile)
                print(N, k, seed, T, k, rt)

                # Set the data

                runtimes[k_idx, N_idx, seed_idx] = rt / (T + 1.)
                if not os.path.exists(time_fname):
                    raise Exception("Warning: profile not found for ", time_fname)
                if not os.path.exists(logfile):
                    raise Exception("Warning: profile not found for ", logfile)

    plot_slice_sns(runtimes, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap, title = title)

def main():
    #algos = ['ucb'] # Could also include 'naive_v1'
    algo = 'ucb'

    #### for MNIST L2, k = 5
    dataset = 'MNIST'
    metric = 'L2'
    Ns = [500]
    ks = [2, 3, 4, 5, 6, 8, 10, 20, 30, 40, 50, 75, 100, 150, 200]
    # ks = [2, 3, 4, 5, 6, 8, 10]#, 20]
    # ks = [30, 40, 50, 75, 100, 150, 200]
    seeds = range(3)
    dir_ = ''
    title = "Scaling with $k$ on MNIST, $d = l_2$"
    show_plots('N', 'weighted_T', Ns, ks, seeds, algo, dataset, metric, dir_, title = title)




if __name__ == '__main__':
    main()
