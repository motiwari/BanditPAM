import numpy as np

def get_dataset(dataset_name, n_data):
    dataset = None
    if dataset_name == "mnist":
        assert n_data<70000, "MNIST has 70000 data points"
        dataset = np.loadtxt('data/MNIST_70k.csv', skiprows=70000-n_data)
    elif dataset_name == "scrna":
        assert n_data<1000, "SCRNA has 1000 data points"
        dataset = np.loadtxt('data/scrna_1k.csv', skiprows=1000-n_data)
    else:
        assert False, "No such dataset"
    return dataset

def get_stat_format(stats):
    mean, std = stats
    return "{:30}".format("{:<.3f} ({:<.3f})".format(mean, std))

def print_summary(stats, dataset_name, n_data, n_medoids):
    print(f"\n[{dataset_name}: {n_data} | k: {n_medoids}]")
    print("".join(get_stat_format(stat) for stat in stats))