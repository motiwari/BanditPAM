'''
Code to automatically parse the logs produced from running experiments.
Used to generate Figures 1(b) - 3(b) of the paper.
'''

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_slice_sns(data_array, fix_k_or_N, Ns, ks, algo, seeds, runtime_plot, title ="Insert title", take_log = True):
    '''
    Plots the number of distance calls vs. N or wall clock runtime vs. N, for
    various seeds, values of k, and datasets.

    Requires the array of distance calls or runtimes for the algo, for each k,
    N, and seed.
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
        if fix_k_or_N == 'k':
            if take_log:
                np_data = np.log10(data_array)
                Nks_plot = np.log10(Nks)
            else:
                np_data = data_array
                Nks_plot = Nks

            sns.set()
            sns.set_style('white')

            fig, ax = plt.subplots(figsize = (7,7))

            # Make a dataframe with the relevant data, for plotting with seaborn
            d = {'N': Nks_plot}
            for seed_idx, seed in enumerate(seeds):
                d["seed_" + str(seed)] = np_data[kN_idx, :, seed_idx]
            df = pd.DataFrame(data = d)
            print(df)

            # Combine the different seeds into 1 column
            melt_df = df.melt('N', var_name='cols', value_name='vals')
            melt_df['N'] += np.random.randn(melt_df['N'].shape[0]) * 0.01 # Add jitter
            sns.scatterplot(x="N", y="vals", data = melt_df, ax = ax, alpha = 0.6)

            # Plot means and error bars
            bars = (1.96/(10**0.5)) * np.std(np_data[kN_idx, :, :], axis = 1) # Slice a specific k, get a 2D array
            means = np.mean(np_data[kN_idx, :, :], axis = 1)
            plt.errorbar(Nks_plot, means, yerr = bars, fmt = '+', capsize = 5, ecolor='black', elinewidth = 1.5, zorder = 100, mec='black', mew = 1.5, label="95% confidence interval")

            # Plot line of best fit
            sl, icpt, r_val, p_val, _ = sp.stats.linregress(Nks_plot, means)
            x_min, x_max = plt.xlim()
            plt.plot([x_min, x_max], [x_min * sl + icpt, x_max * sl + icpt], color='black', label='Linear fit, slope=%0.3f'%(sl))

            print("Slope is:", sl, "Intercept is:", icpt)

            # Manually modify legend labels for prettiness
            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles[::-1], labels[::-1], loc="upper left")

        elif fix_k_or_N == 'N':
            raise Exception("Fixing N and plotting vs. k not yet supported")

        # Set axis labels based on whether we're taking the log
        # and whether we're plotting runtime or distance computations
        xlabel_str = ""
        ylabel_str = ""
        if take_log:
            xlabel_str = "$\log_{10}(n)$"
            ylabel_str += "$\log_{10}$("
        else:
            xlabel_str = "$n$"

        if runtime_plot:
            ylabel_str += "runtime (s)"
        else:
            ylabel_str += "average # of distance computations per step"

        if take_log:
            ylabel_str += ")"

        plt.ylabel(ylabel_str)
        plt.xlabel(xlabel_str)

        plt.title(title)
        plt.savefig('figures/' + title.replace("$", '') + '.pdf')

def get_dist_comps(logfile):
    '''
    Get the number of distance computations performed in an experiment, from parsing the
    logfile
    '''

    with open(logfile, 'r') as fin:
        line = fin.readline()
        while line[:22] != 'Distance Computations:':
            line = fin.readline()

        dist_comps = int(line.split(' ')[-1])
    return dist_comps

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

def show_plots(fix_k_or_N, Ns, ks, seeds, algo, dataset, metric, dir_, runtime_plot, title = "Insert title"):
    '''
    A function which mines the number of distance calls or runtime for each
    experiment, from the dumped logs. Creates a numpy array with the distance
    call or runtime counts.

    It does this by:
        - first, identifying the filenames where the experiment
            logs are stored (the logfile is used for the number of swap
            steps and the timefile is used for the runtime)
        - searching each logfile for the number of distance
            calls (if not making a runtime plot)
        - weighting the distance calls or runtime between the swap steps
    '''
    data_array = np.zeros((len(ks), len(Ns), len(seeds)))
    log_prefix = 'logs/' + dir_ + '/L-'

    # Gather data
    assert algo in ['bfp'], "Bad algo input"
    for N_idx, N in enumerate(Ns):
        for k_idx, k in enumerate(ks):
            for seed_idx, seed in enumerate(seeds):
                # Get the number of swaps
                logfile = log_prefix + algo + '-k-' + str(k) + \
                          '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                if not os.path.exists(logfile):
                    raise Exception("Warning: logfile not found for ", logfile)

                T = get_swap_T(logfile)

                if runtime_plot:
                    # Get the runtime
                    time_prefix = 'logs/' + dir_ + '/t-'

                    time_fname = time_prefix + algo + '-k-' + str(k) + \
                        '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                    if not os.path.exists(time_fname):
                        raise Exception("Warning: timefile not found for ", time_fname)

                    rt = get_runtime(time_fname)
                    print(N, k, seed, T, rt)

                    # Set the data
                    data_array[k_idx, N_idx, seed_idx] = rt / T
                else:
                    # Get the number of distance computations
                    dist_comps = get_dist_comps(logfile)
                    print(T, dist_comps, k)

                    # Set the data
                    data_array[k_idx][N_idx][seed_idx] += dist_comps / T

    plot_slice_sns(data_array, fix_k_or_N, Ns, ks, algo, seeds, runtime_plot, title = title)

def main():
    '''
    Make a plot showing either the number of distance computations or the
    runtime vs. N for BanditFasterPAM.
    '''
    algo = 'bfp'
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    seeds = range(42, 72)
    runtime_plot = True

    ######## Figure 1 (b): CIFAR, L1, k = 2
    # dataset = 'CIFAR'
    # metric = 'L1'
    # ks = [2]
    # dir_ = 'CIFAR_L1_k2_paper'
    # title = "CIFAR, $d = l_1$, $k = 2$"
    # runtime_plot = False

    ######## Figure 2 (a): MNIST, L2, k = 3
    dataset = 'MNIST'
    metric = 'L2'
    ks = [3]
    dir_ = 'MNIST_L2_k3_paper'
    title = "MNIST, $d = l_2, k = 3$"

    ######## Figure 2 (b): MNIST, L2, k = 5
    # dataset = 'MNIST'
    # metric = 'L2'
    # ks = [5]
    # dir_ = 'MNIST_L2_k5_paper'
    # title = "MNIST, $d = l_2, k = 5$"

    ######## Figure 3 (a): MNIST, cosine, k = 3
    # dataset = 'MNIST'
    # metric = 'cosine'
    # ks = [3]
    # dir_ = 'MNIST_COSINE_k3_paper'
    # title = "MNIST, $d =$ cosine, $k = 3$"

    ######## Figure 3 (b): SCRNA, L1, k = 3
    # dataset = 'SCRNA'
    # metric = 'L1'
    # ks = [3]
    # dir_ = 'SCRNA_L1_k3_paper'
    # title = "scRNA, $d = l_1, k = 3$"

    show_plots('k', Ns, ks, seeds, algo, dataset, metric, dir_, runtime_plot, title = title)


if __name__ == '__main__':
    main()
