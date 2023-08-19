'''
Compare the losses of BanditFasterPAM, FasterPAM, and PAM.
Used to generate Figure 1(a) of the paper.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def get_file_loss(file_):
    '''
    Get the final loss of an experiment from the logfile
    '''
    # the final loss is on the 4th line of the logfile
    num_lines = 4

    with open(file_, 'r') as fin:
        line_idx = 0
        while line_idx < num_lines:
            line_idx += 1
            line = fin.readline()

        final_loss = line.split(' ')[-1]
        return float(final_loss)

def verify_logfiles():
    '''
    Verifies that BanditFasterPAM returns the same SWAP medoid assignments as
    FasterPAM, by parsing the logfiles. Note that the same seed must be used for
    both algorithms to ensure that the uniform random sampling is the same.
    '''
    # Currently, the loss plot is the only plot where both BanditFasterPAM and
    # FasterPAM are run on the same dataset
    parent_dirs = [
        'logs/Loss_plots_paper',
    ]
    for parent_dir in parent_dirs:
        bfp_logfiles = [os.path.join(parent_dir, x) for x in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, x)) and x != '.DS_Store' and x[:5] == 'L-bfp']
        for bfp_lfile in sorted(bfp_logfiles):
            fp_lfile = bfp_lfile.replace('bfp', 'fp')
            if not os.path.exists(fp_lfile):
                print("Warning: no FasterPAM experiment", fp_lfile)
            else:
                disagreement = False
                with open(bfp_lfile, 'r') as fin1:
                    with open(fp_lfile, 'r') as fin2:
                        l1_1 = fin1.readline().strip().split(",")
                        l1_2 = fin1.readline().strip().split(",")

                        l2_1 = fin2.readline().strip().split(",")
                        l2_2 = fin2.readline().strip().split(",")

                        # compare the swaps performed
                        if sorted(l1_2) != sorted(l2_2):
                            disagreement = True

                if disagreement:
                    print("\n")
                    print(sorted(l1_2))
                    print(sorted(l2_2))
                    print("ERROR: Results for", bfp_lfile, fp_lfile, "disagree!!")
                else:
                    print("OK: Results for", bfp_lfile, fp_lfile, "agree")

def make_plots():
    '''
    Make a plot showing the relative losses of BanditPAM and
    FasterPAM, normalized to PAM's loss. Used for Figure 1(a) of the paper.
    '''

    loss_dir = 'logs/Loss_plots_paper/'

    algos = ['naive_v1', 'bfp', 'fp']
    seeds = range(10)
    Ns = [5000, 7500, 10000]
    k = 5

    alg_to_legend = {
        'naive_v1' : 'PAM',
        'bfp' : 'BanditFasterPAM',
        'fp': 'FasterPAM',
    }

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
                filename = loss_dir + 'L-' + algo + '-k-' + str(k) + '-N-' + str(N) + '-s-' + str(seed + 42) + '-d-MNIST-m-L2'

                if not os.path.exists(filename):
                    raise Exception("Warning: logfile not found for ", filename)

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
    # verify that BanditFasterPAM and FasterPAM make the same swaps
    verify_logfiles()
    print("FILES VERIFIED\n\n")
    make_plots()
