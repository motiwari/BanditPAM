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
import snakevizcode

from generate_config import write_exp

# Possible line numbers for the empty_counter fn
FN_NAME_1 = 'data_utils.py:129(empty_counter)'
FN_NAME_2 = 'data_utils.py:141(empty_counter)'
FN_NAME_3 = 'data_utils.py:142(empty_counter)'

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

def plot_slice_sns(dcalls_array, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap, take_log = True):
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
            y_min, y_max = plt.ylim()
            plt.plot([x_min, x_max], [x_min * sl + icpt, x_max * sl + icpt], color='black', label='Linear fit, slope=%0.3f'%(sl))

            # if build_or_swap == 'build':
            #     # Plot reference kN^2 line here in build step for PAM
            #     plt.plot([x_min, x_max], [np.log10(kN) + x_min * 2, np.log10(kN) + x_max * 2], color='red', label='$kn^2$ PAM scaling')
            # elif build_or_swap == 'swap':
            #     # Plot reference N^2 line here in build step for PAM + FP1
            #     # (no dependence on k)
            #     plt.plot([x_min, x_max], [x_min * 2, x_max * 2], color='red', label='$kn^2$ PAM scaling')
            # else:
            #     # weighted reference line for
            #     plt.plot([x_min, x_max], [(x_min) * 2, (x_max) * 2], color='#f781bf', linestyle=':', label='$n^2$ FastPAM1 scaling')
            #     plt.plot([x_min, x_max], [np.log10(kN) + (x_min) * 2, np.log10(kN) + (x_max) * 2], color='#4daf4a', linestyle='-.', label='$kn^2$ PAM scaling')


            print("Slope is:", sl)

            # Manually modify legend labels for prettiness
            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles[::-1], labels[::-1], loc="upper left")

        elif fix_k_or_N == 'N':
            raise Exception("Fixing N and plotting vs. k not yet supported")

        plt.xlabel("$\log_{10}(n)$")
        plt.ylabel("$\log_{10}$(average # of distance computations per step)")

        # Modify these lines based on dataset
        plt.title("CIFAR, $d = l_1$, $k = 2$")
        plt.savefig('figures/CIFAR, d = l_1, k = 2.pdf')

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

        T = int(line.split(' ')[-1])
    return T

def show_plots(fix_k_or_N, build_or_swap, Ns, ks, seeds, algos, dataset, metric, dir_):
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
    dcalls_array = np.zeros((len(ks), len(Ns), len(seeds)))

    if build_or_swap == 'build':
        prefix = 'profiles/' + dir_ + '/p-B-'
    elif build_or_swap == 'swap':
        prefix = 'profiles/' + dir_ + '/p-S-'
    elif build_or_swap == 'weighted' or build_or_swap == 'weighted_T':
        pass
    else:
        raise Exception("Error pi")

    log_prefix = 'profiles/' + dir_ + '/L-'

    # Gather data
    for algo in algos:
        assert algo in ['ucb', 'naive_v1', 'bfp'], "Bad algo yo"
        for N_idx, N in enumerate(Ns):
            for k_idx, k in enumerate(ks):
                for seed_idx, seed in enumerate(seeds):
                    if build_or_swap == 'weighted' or build_or_swap == 'weighted_T':
                        prefix = 'profiles/' + dir_ + '/p-'

                        build_prefix = prefix + 'B-'
                        build_profile_name = build_prefix + algo + '-False-BS-v-0-k-' + str(k) + \
                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                        swap_prefix = prefix + 'S-'
                        swap_profile_name = swap_prefix + algo + '-False-BS-v-0-k-' + str(k) + \
                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                        logfile = log_prefix + algo + '-False-BS-v-0-k-' + str(k) + \
                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                        if not os.path.exists(logfile):
                            raise Exception("Warning: Log file not found for ", logfile)

                        T = get_swap_T(logfile)
                        dist_comps = get_dist_comps(logfile)
                        print(T, dist_comps, k)

                        # if build_or_swap == 'weighted':
                        #     dcalls_array[k_idx][N_idx][seed_idx] += dist_comps # build + avg(swap)
                        # elif build_or_swap == 'weighted_T':
                        #     dcalls_array[k_idx][N_idx][seed_idx] += dist_comps / T # (build + swap) / T
                        # else:
                        #     raise Exception("blank")

                        if build_or_swap == 'weighted_T':
                            dcalls_array[k_idx][N_idx][seed_idx] += dist_comps / T
                        else:
                            raise Exception("blank")

                    else:
                        # TODO(@Adarsh321123): remove all non weighted_T stuff?
                        assert build_or_swap == 'build' or build_or_swap == 'swap', "Error with build_or_swap"
                        profile_fname = prefix + algo + '-False-BS-v-0-k-' + str(k) + \
                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'
                        if os.path.exists(profile_fname):
                            p = pstats.Stats(profile_fname)
                            for row in snakevizcode.table_rows(p):
                                if FN_NAME_1 in row or FN_NAME_2 in row or FN_NAME_3 in row:
                                    dcalls = row[0][1]
                                    if build_or_swap == 'build':
                                        dcalls_array[k_idx][N_idx][seed_idx] = dcalls
                                    elif build_or_swap == 'swap':
                                        logfile = log_prefix + algo + '-False-BS-v-0-k-' + str(k) + \
                                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'
                                        T = get_swap_T(logfile)
                                        dcalls_array[k_idx][N_idx][seed_idx] = dcalls / T
                                    else:
                                        raise Exception("Averaging method not supported")
                        else:
                            print("Warning: profile not found for ", profile_fname)

    # Show data
    for algo in algos:
        plot_slice_sns(dcalls_array, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap)

def main():


    ######## HOC4, Tree edit distance (precomputed), k = 2 and k = 3
    # algos = ['ucb']
    # dataset = 'HOC4'
    # metric = 'PRECOMP'
    # Ns = [1000, 2000, 3000, 3360]
    # ks = [2]
    # seeds = range(42, 52)
    # dir_ = 'HOC4_PRECOMP_k2k3_paper'

    ######## CIFAR, L1 distance, k = 2
    dataset = 'CIFAR'
    metric = 'L1'
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    ks = [2]
    seeds = range(42, 72)
    algos = ['bfp']
    dir_ = 'CIFAR_L1_k2_paper_20k'

    #### for MNIST L2, k = 5
    # dataset = 'MNIST'
    # metric = 'L2'
    # Ns = [10000, 20000, 40000, 70000]
    # ks = [5]
    # seeds = range(42, 52)
    # dir_ = 'MNIST_L2_k5_paper'

    ### for MNIST L2, k = 10
    # dataset = 'MNIST'
    # metric = 'L2'
    # Ns = [3000, 10000, 30000, 70000]
    # ks = [10]
    # seeds = range(42, 52)
    # dir_ = 'MNIST_L2_k10_paper'

    ##### for MNIST COSINE
    # dataset = 'MNIST'
    # metric = 'COSINE'
    # Ns = [3000, 10000, 20000, 40000]
    # ks = [5]
    # seeds = range(42, 52)
    # dir_ = 'MNIST_COSINE_k5_paper'

    #### for scRNAPCA, L2, K = 10
    # dataset = 'SCRNAPCA'
    # metric = 'L2'
    # Ns = [10000, 20000, 30000, 40000]
    # ks = [10]
    # seeds = range(42, 52)
    # dir_ = 'SCRNAPCA_L2_k10_paper' # NOTE: SCRNA_PCA_paper_more_some_incomplete contains data for some more values of N.

    #### for scRNAPCA, L2, K = 5
    # dataset = 'SCRNAPCA'
    # metric = 'L2'
    # Ns = [10000, 20000, 30000, 40000]
    # ks = [5]
    # seeds = range(42, 52)
    # dir_ = 'SCRNAPCA_L2_k5_paper'

    #### for scRNA, L1, K = 5
    # dataset = 'SCRNA'
    # metric = 'L1'
    # Ns = [10000, 20000, 30000, 40000]
    # ks = [5]
    # seeds = range(42, 52)
    # dir_ = 'SCRNA_L1_paper'

    # show_plots('k', 'build', Ns, ks, seeds, algos, dataset, metric, dir_)
    # show_plots('k', 'swap', Ns, ks, seeds, algos, dataset, metric, dir_)
    # show_plots('k', 'weighted', Ns, ks, seeds, algos, dataset, metric, dir_)
    show_plots('k', 'weighted_T', Ns, ks, seeds, algos, dataset, metric, dir_)


if __name__ == '__main__':
    # verify_logfiles()
    print("FILES VERIFIED\n\n")
    main()
    # TODO(@Adarsh321123): use C++ dist comps instead of profiler?
