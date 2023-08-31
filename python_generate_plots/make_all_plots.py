"""
Convenience code to automatically generate all plots used in the
BanditFasterPAM AAAI submission.
"""

import run_experiments
import make_loss_plots
import parse_logs


def make_exps(ks, seeds, Ns, algos, dataset, metric):
    # Due to issues with writing and reading being out of sync, which causes
    # the wrong experiment to be run at times, we directly make the
    # dictionary of experiments to run.
    experiments = []
    for k in ks:
        for seed in seeds:
            for N in Ns:
                for algo in algos:
                    experiments.append(
                        [algo, k, N, 42 + seed, dataset, metric]
                    )
    return experiments


def main():
    ######## Figure 1 (a): loss plots
    dataset = "MNIST"
    metric = "L2"
    Ns = [5000, 7500, 10000]
    ks = [5]
    algos = ["bfp", "fp", "naive_v1"]
    seeds = range(10)
    experiments = make_exps(ks, seeds, Ns, algos, dataset, metric)

    dir_name = "Loss_plots_paper_manual"
    run_experiments.main(dir_name, experiments)

    loss_dir_name = f"logs/{dir_name}/"
    make_loss_plots.make_plots(loss_dir_name)

    ######## Figure 1 (b): CIFAR, L1, k = 2
    dataset = "CIFAR"
    metric = "L1"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    ks = [2]
    algos = ["bfp"]
    seeds = range(30)
    experiments = make_exps(ks, seeds, Ns, algos, dataset, metric)

    dir_name = "CIFAR_L1_k2_paper_manual"
    run_experiments.main(dir_name, experiments)

    algo = "bfp"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    seeds = range(42, 72)
    dataset = "CIFAR"
    metric = "L1"
    ks = [2]
    title = "CIFAR, $d = l_1$, $k = 2$"
    runtime_plot = False
    parse_logs.main(
        algo, Ns, seeds, dataset, metric, ks, title, runtime_plot, dir_name
    )

    ######## Figure 2 (a): MNIST, L2, k = 3
    dataset = "MNIST"
    metric = "L2"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    ks = [3]
    algos = ["bfp"]
    seeds = range(30)
    experiments = make_exps(ks, seeds, Ns, algos, dataset, metric)

    dir_name = "MNIST_L2_k3_paper_manual"
    run_experiments.main(dir_name, experiments)

    algo = "bfp"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    seeds = range(42, 72)
    runtime_plot = True
    dataset = "MNIST"
    metric = "L2"
    ks = [3]
    title = "MNIST, $d = l_2, k = 3$"
    parse_logs.main(
        algo, Ns, seeds, dataset, metric, ks, title, runtime_plot, dir_name
    )

    ######## Figure 2 (b): MNIST, L2, k = 5
    dataset = "MNIST"
    metric = "L2"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    ks = [5]
    algos = ["bfp"]
    seeds = range(30)
    experiments = make_exps(ks, seeds, Ns, algos, dataset, metric)

    dir_name = "MNIST_L2_k5_paper_manual"
    run_experiments.main(dir_name, experiments)

    algo = "bfp"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    seeds = range(42, 72)
    runtime_plot = True
    dataset = "MNIST"
    metric = "L2"
    ks = [5]
    title = "MNIST, $d = l_2, k = 5$"
    parse_logs.main(
        algo, Ns, seeds, dataset, metric, ks, title, runtime_plot, dir_name
    )

    ######## Figure 3 (a): MNIST, cosine, k = 3
    dataset = "MNIST"
    metric = "cosine"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    ks = [3]
    algos = ["bfp"]
    seeds = range(30)
    experiments = make_exps(ks, seeds, Ns, algos, dataset, metric)

    dir_name = "MNIST_COSINE_k3_paper_manual"
    run_experiments.main(dir_name, experiments)

    algo = "bfp"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    seeds = range(42, 72)
    runtime_plot = True
    dataset = "MNIST"
    metric = "cosine"
    ks = [3]
    title = "MNIST, $d =$ cosine, $k = 3$"
    parse_logs.main(
        algo, Ns, seeds, dataset, metric, ks, title, runtime_plot, dir_name
    )

    ######## Figure 3 (b): SCRNA, L1, k = 3
    dataset = "SCRNA"
    metric = "L1"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    ks = [3]
    algos = ["bfp"]
    seeds = range(30)
    experiments = make_exps(ks, seeds, Ns, algos, dataset, metric)

    dir_name = "SCRNA_L1_k3_paper_manual"
    run_experiments.main(dir_name, experiments)

    algo = "bfp"
    Ns = [5000, 7500, 10000, 12500, 15000, 17500, 20000]
    seeds = range(42, 72)
    runtime_plot = True
    dataset = "SCRNA"
    metric = "L1"
    ks = [3]
    title = "scRNA, $d = l_1, k = 3$"
    parse_logs.main(
        algo, Ns, seeds, dataset, metric, ks, title, runtime_plot, dir_name
    )


if __name__ == "__main__":
    main()
