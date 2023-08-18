'''
Compare the losses of BanditPAM and various baselines: PAM, FastPAM, EM, CLARANS
Used to generate Figure 1(a) of the paper.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def get_file_loss(file_):
    '''
    Get the final loss of an experiment from the logfile
    '''
    if 'bfp' in file_ or 'fp' in file_ or 'naive_v1' in file_:
        num_lines = 4
    else:
        num_lines = 2

    with open(file_, 'r') as fin:
        line_idx = 0
        while line_idx < num_lines:
            line_idx += 1
            line = fin.readline()

        final_loss = line.split(' ')[-1]
        return float(final_loss)

def get_swaps(file_):
    '''
    Get the number of swaps performed in an experiment from the logfile.
    '''

    with open(file_, 'r') as fin:
        swaps = []
        line = fin.readline()
        while line.strip() != 'Swap Logstring:': # Need to get past the 'swap:' line in build logstring
            line = fin.readline()

        while line.strip() != 'swap:':
            line = fin.readline()

        line = fin.readline()
        while line:
            medoids_swapped = line.split(' ')[-1].strip()
            swaps.append(medoids_swapped)
            line = fin.readline()

        last_old_medoid = medoids_swapped.split(',')[0]
        last_new_medoid = medoids_swapped.split(',')[1].strip()
        assert last_old_medoid == last_new_medoid, "The last swap should try to swap a medoid with itself"
        return swaps

def get_build_meds(file_):
    '''
    Get the medoids returned by just the BUILD step for an experiment, from its
    logfile.
    '''

    with open(file_, 'r') as fin:
        line = fin.readline()
    return line.strip()


def get_swap_meds(file_):
    '''
    Get the final medoids returned by the SWAP step for an experiment, from
    its logfile.
    '''

    with open(file_, 'r') as fin:
        line = fin.readline()
        line = fin.readline()
    return line.strip()

def verify_optimization_paths():
    '''
    Verifies that BanditPAM followed the exact same optimization path as PAM, by
    parsing the logfiles of both experiments.
    '''

    loss_dir = 'profiles/Loss_plots_paper/'

    algos = ['naive_v1', 'ucb']
    seeds = range(10)
    Ns = [500, 1000, 1500, 2000, 2500, 3000]
    k = 5

    for N_idx, N in enumerate(Ns):
        for seed_idx, seed in enumerate(seeds):
            ucb_filename = loss_dir + 'L-ucb-False-BS-v-0-k-' + str(k) + '-N-' + str(N) + '-s-' + str(seed + 42) + '-d-MNIST-m-L2-w-'
            naive_filename = loss_dir + 'L-naive_v1-False-BS-v-0-k-' + str(k) + '-N-' + str(N) + '-s-' + str(seed + 42) + '-d-MNIST-m-L2-w-'

            ucb_built = get_build_meds(ucb_filename)
            ucb_swapped = get_swap_meds(ucb_filename)
            ucb_swaps = get_swaps(ucb_filename)

            naive_built = get_build_meds(naive_filename)
            naive_swapped = get_swap_meds(naive_filename)
            naive_swaps = get_swaps(naive_filename)

            if ucb_built != naive_built:
                print("Build medoids disagree on " + str(N) + ',' + str(seed))
                print(naive_built)
                print(ucb_built)

            if ucb_swapped != naive_swapped:
                print("Build medoids disagree on " + str(N) + ',' + str(seed))
                print(naive_swapped)
                print(ucb_swapped)

            if ucb_swaps != naive_swaps:
                print("Build medoids disagree on " + str(N) + ',' + str(seed))
                print(naive_swaps)
                print(ucb_swaps)

def get_FP_loss(N, seed):
    '''
    Get the losses from running FastPAM. These were manually obtained by using
    the ELKI GUI implementation of FastPAM.
    '''

    with open('ELKI/manual_fastpam_losses.txt', 'r') as fin:
        prefix = "N=" + str(N) + ",seed=" + str(seed + 42)+":"
        line = fin.readline()
        while line[:len(prefix)] != prefix:
            line = fin.readline()

        fp_loss = float(line.split(':')[-1])/N
        return fp_loss

def make_plots():
    '''
    Make a plot showing the relative losses of BanditPAM, EM, CLARANS, and
    FastPAM, normalized to PAM's loss. Used for Figure 1(a) of the paper.
    '''

    loss_dir = 'profiles/Loss_plots_paper_20k/'

    algos = ['naive_v1', 'bfp', 'fp']
    # algos = ['bfp', 'fp']
    seeds = range(10)
    # Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    Ns = [5000, 7500, 10000]
    k = 5

    mult_jitter = 20

    alg_to_legend = {
        'naive_v1' : 'PAM',
        'bfp' : 'BanditFasterPAM',
        'fp': 'FasterPAM',
    }

    ADD_JITTER = 75
    alg_to_add_jitter = {
        'naive_v1' : 0,
        'bfp' : 0,
        'fp' : 0,
    }

    alg_color = {
        'naive_v1' : 'orange',
        'bfp' : 'b',
        'fp' : 'r',
    }

    alg_zorder = {
        'naive_v1' : 0,
        'bfp' : 2,
        'fp' : 1,
    }

    losses = np.zeros((len(Ns), len(algos) + 1, len(seeds)))

    for N_idx, N in enumerate(Ns):
        for algo_idx, algo in enumerate(algos):
            for seed_idx, seed in enumerate(seeds):
                filename = loss_dir + 'L-' + algo + '-False-BS-v-0-k-' + str(k) + '-N-' + str(N) + '-s-' + str(seed + 42) + '-d-MNIST-m-L2-w-'
                losses[N_idx, algo_idx, seed_idx] = get_file_loss(filename)


    # Normalize losses to PAM
    for N_idx, N in enumerate(Ns):
        for seed_idx, seed in enumerate(seeds):
            naive_value = losses[N_idx, 0, seed_idx]
            losses[N_idx, :, seed_idx] /= naive_value

    sns.set()
    sns.set_style('white')
    fig, ax = plt.subplots(figsize = (6, 5))
    plt.ylim(0.995, 1.07)
    plt.xlim(4000, 11000)
    ax.axhline(1, ls='-.', color = 'black', zorder = -100, linewidth = 0.4)

    for algo_idx, algorithm in enumerate(algos):
        if algorithm == 'naive_v1': continue

        this_color = alg_color[algorithm]
        this_label = alg_to_legend[algorithm]
        this_jitter = alg_to_add_jitter[algorithm]
        this_zorder = alg_zorder[algorithm]

        d = {'N': Ns}
        for seed_idx, seed in enumerate(seeds):
            d["seed_" + str(seed)] = losses[:, algo_idx, seed_idx]
        df = pd.DataFrame(data = d)

        melt_df = df.melt('N', var_name='cols', value_name='vals')
        melt_df['N'] += np.random.randn(melt_df['N'].shape[0]) + this_jitter # Add jitter


        bars = (1.96/(10**0.5)) * np.std(losses[:, algo_idx, :], axis = 1) # Slice a specific algo, get a 2D array
        means = np.mean(losses[:, algo_idx, :], axis = 1)
        print(algorithm, this_color, this_label, this_jitter)
        plt.plot(np.array(Ns) + this_jitter, means, color=this_color, zorder=this_zorder, linewidth = 2)
        plt.errorbar(np.array(Ns) + this_jitter, means, yerr = bars, fmt = '+', capsize = 5, ecolor = this_color, elinewidth = 1.5, zorder = this_zorder, mec=this_color, mew = 1.5, label = this_label)

    plt.xlabel("$n$")
    plt.ylabel(r'Final Loss Normalized to PAM ($L/L_{PAM}$)')
    plt.title("$L/L_{PAM}$ vs. $n$ (MNIST, $d = l_2, k = 5$)")
    plt.legend()
    plt.savefig('figures/loss_plot.pdf')


if __name__ == "__main__":
    loss_dir = 'profiles/Loss_plots_paper_20k/'
    # verify_optimization_paths()
    make_plots()
