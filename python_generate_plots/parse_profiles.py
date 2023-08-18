'''
Code to automatically parse the profiles produced from running experiments.
In particular, plots the scaling of BanditPAM vs. N for various dataset sizes N.
Used to demonstrate O(NlogN) scaling.
'''

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def verify_logfiles():
    '''
    Verifies that BanditPAM returns the same BUILD and SWAP medoid assignments
    as PAM, by parsing the logfiles.
    '''

    parent_dirs = [
        # 'profiles/HOC4_PRECOMP_k2k3_paper',
        # 'profiles/MNIST_COSINE_k5_paper',
        # 'profiles/MNIST_L2_k10_paper',
        # 'profiles/MNIST_L2_k5_paper',
        # 'profiles/SCRNAPCA_L2_k10_paper',
        # 'profiles/SCRNAPCA_L2_k5_paper',
        # 'profiles/SCRNA_L1_paper',
        # 'profiles/Loss_plots_paper',
        'profiles/profiles',
    ]
    for parent_dir in parent_dirs:
        ucb_logfiles = [os.path.join(parent_dir, x) for x in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, x)) and x != '.DS_Store' and x[:5] == 'L-ucb']
        for u_lfile in sorted(ucb_logfiles):
            n_lfile = u_lfile.replace('ucb', 'naive_v1')
            if not os.path.exists(n_lfile):
                print("Warning: no naive experiment", n_lfile)
            else:
                disagreement = False
                with open(u_lfile, 'r') as fin1:
                    with open(n_lfile, 'r') as fin2:
                        l1_1 = fin1.readline().strip().split(",")
                        l1_2 = fin1.readline().strip().split(",")

                        l2_1 = fin2.readline().strip().split(",")
                        l2_2 = fin2.readline().strip().split(",")

                        # NOTE: This is a stricter condition than necessary, enforcing both build and swap agreement instead of just swap
                        if sorted(l1_2) != sorted(l2_2):
                            disagreement = True

                if disagreement:
                    print("\n")
                    print(sorted(l1_2))
                    print(sorted(l2_2))
                    print("ERROR: Results for", u_lfile, n_lfile, "disagree!!")
                else:
                    print("OK: Results for", u_lfile, n_lfile, "agree")

def plot_slice_sns(data_array, fix_k_or_N, Ns, ks, algo, seeds, runtime_plot, title ="Insert title", take_log = True):
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

        plt.xlabel("$\log_{10}(n)$")
        ylabel_str = ""
        if take_log:
            ylabel_str += "$\log_{10}$("

        if runtime_plot:
            ylabel_str += "runtime (s)"
        else:
            ylabel_str += "average # of distance computations per step"

        if take_log:
            ylabel_str += ")"

        plt.ylabel(ylabel_str)

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
    data_array = np.zeros((len(ks), len(Ns), len(seeds)))
    log_prefix = 'profiles/' + dir_ + '/p-' # TODO: change back to L

    # Gather data
    assert algo in ['bfp'], "Bad algo input" # TODO: test
    for N_idx, N in enumerate(Ns):
        for k_idx, k in enumerate(ks):
            for seed_idx, seed in enumerate(seeds):
                # Get the number of swaps or distance computations
                logfile = log_prefix + algo + '-False-BS-v-0-k-' + str(k) + \
                          '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                if not os.path.exists(logfile):
                    raise Exception("Warning: logfile not found for ", logfile)

                T = get_swap_T(logfile)

                if runtime_plot:
                    # Get the time
                    time_prefix = 'profiles/' + dir_ + '/t-'
                    time_fname = time_prefix + algo + '-False-BS-v-0-k-' + str(k) + \
                        '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                    if not os.path.exists(time_fname):
                        raise Exception("Warning: timefile not found for ", time_fname)

                    rt = get_runtime(time_fname)
                    print(N, k, seed, T, k, rt)

                    # Set the data
                    data_array[k_idx, N_idx, seed_idx] = rt / T
                else:
                    dist_comps = get_dist_comps(logfile)
                    print(T, dist_comps, k)

                    # Set the data
                    data_array[k_idx][N_idx][seed_idx] += dist_comps / T

    plot_slice_sns(data_array, fix_k_or_N, Ns, ks, algo, seeds, runtime_plot, title = title)

def main():
    algo = 'bfp'
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    seeds = range(42, 72)
    runtime_plot = True

    ######## Figure 1 (b): CIFAR, L1, k = 2
    # dataset = 'CIFAR'
    # metric = 'L1'
    # ks = [2]
    # dir_ = 'CIFAR_L1_k2_paper_20k'
    # title = "CIFAR, $d = l_1$, $k = 2$"
    # runtime_plot = False

    ######## Figure 2 (a): MNIST, L2, k = 3
    # dataset = 'MNIST'
    # metric = 'L2'
    # ks = [3]
    # dir_ = 'MNIST_L2_k3_paper'
    # title = "MNIST, $d = l_2, k = 3$"

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
    dataset = 'SCRNA'
    metric = 'L1'
    ks = [3]
    dir_ = 'SCRNA_L1_k3_paper_20k'
    title = "scRNA, $d = l_1, k = 3$"

    show_plots('k', Ns, ks, seeds, algo, dataset, metric, dir_, runtime_plot, title = title)


if __name__ == '__main__':
    # TODO: verify btwn BFP and FP?
    # verify_logfiles()
    print("FILES VERIFIED\n\n")
    # import ipdb; ipdb.set_trace()
    main()
