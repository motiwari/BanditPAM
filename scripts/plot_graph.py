from typing import List
import os
from matplotlib import pyplot as plt
import glob
from pandas import read_csv
import numpy as np

from constants import (
    # datasets
    MNIST,
    SCRNA,

    # algorithms
    ALL_BANDITPAMS,

    # experiment settings
    NUM_DATA,
    NUM_MEDOIDS,
    RUNTIME,
    SAMPLE_COMPLEXITY,

    # utils
    ALG_TO_COLOR
)


def extract_algorithm_from_filename(filename):
    for algorithm in ALL_BANDITPAMS:
        if algorithm in filename:
            return algorithm


def translate_experiment_setting(setting):
    """
    Translate a setting into a human-readable format
    For example, "k5" becomes "Num medoids: 5"
    """
    if "k" in setting:
        return f"Num medoids : {setting[1:]}"
    elif "n" in setting:
        return f"Num data : {setting[1:]}"
    else:
        assert False, "Invalid setting"


def create_scaling_plots(
        datasets: List[str] = [],
        algorithms: List[str] = [],
        x_axes=NUM_DATA,
        y_axes=RUNTIME,
        is_fit: bool = False,
        is_logspace_x: bool = False,
        is_logspace_y: bool = False,
        dir_name: str = None,
):
    """
    Plot the scaling experiments from the data stored in the logs file.

    :param caching_type: either "", "(naive_cache)", "(PI_cache)"
    :param datasets: the datasets that you want to plot. If empty, warning is triggered.
    :param algorithms: the algorithms that you want to plot. If empty, warning is triggered.
    :param ind_variables: Independent variables (N or D)
    :param include_error_bar: shows the standard deviation
    :param is_fit: whether to fit the plots or not
    :param is_logspace_x: whether to plot x-axis in logspace or not
    :param is_logspace_y: whether to plot y-axis in logspace or not
    :param is_plot_accuracy: whether to annotate accuracy in the graph
    :param dir_name: directory name of log files
    """
    if len(datasets) == 0:
        raise Exception("At least one dataset must be specified")

    if len(algorithms) == 0:
        raise Exception("At least one algorithm must be specified")

    for x_axis in x_axes:
        # get log csv files
        if dir_name is None:
            parent_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir_name = "scaling_with_n_cluster" if x_axis == NUM_DATA else "scaling_with_k_cluster"
            log_dir = os.path.join(parent_dir, "experiments", "logs", log_dir_name)
        else:
            log_dir = dir_name

        for dataset in datasets:
            csv_files = glob.glob(os.path.join(log_dir, f"*{dataset}*"))

            # list available settings by parsing the log file names
            # for example, "settings" returns ["k5", "k10"] if the experiment used "k = 5" and "k = 10" settings
            settings = list(set([file.split("_")[-1][:-4] for file in csv_files]))

            for setting in settings:
                for y_axis in y_axes:

                    title = f"Scaling Baselines: {dataset} ({translate_experiment_setting(setting)})"

                    for f in csv_files:
                        if str(setting) not in f:
                            continue

                        alg_name = extract_algorithm_from_filename(f)
                        data = read_csv(f)

                        # only get relevant data
                        if f.find(dataset) < 0 or alg_name not in algorithms:
                            continue

                        x = data[x_axis].tolist()
                        if y_axis is SAMPLE_COMPLEXITY:
                            y = data["total_complexity_with_misc"].tolist()
                        else:
                            y = data["total_runtime"].tolist()
                        x, y = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))  # sort

                        if is_logspace_x:
                            x = np.log(x)

                        if is_logspace_y:
                            y = np.log(y)

                        plt.scatter(
                            x,
                            y,
                            color=ALG_TO_COLOR[alg_name],
                            label=alg_name,
                        )
                        plt.plot(x, y, color=ALG_TO_COLOR[alg_name])

                        # Sort the legend entries (labels and handles) by labels
                        handles, labels = plt.gca().get_legend_handles_labels()
                        labels, handles = zip(
                            *sorted(zip(labels, handles), key=lambda t: t[0])
                        )
                        plt.legend(handles, labels, loc="upper left")

                        plt.title(title)

                        xlabel = (
                            "Number of data" if x_axis == NUM_DATA else "Number of medoids"
                        )
                        if is_logspace_x:
                            xlabel = f"ln({xlabel})"

                        ylabel = "Sample Complexity" if y_axis == SAMPLE_COMPLEXITY else "Runtime"
                        if is_logspace_y:
                            ylabel = f"ln({ylabel})"

                        plt.xlabel(xlabel)
                        plt.ylabel(ylabel)

                    plt.show()


if __name__ == "__main__":
    create_scaling_plots(datasets=[MNIST],
                         algorithms=ALL_BANDITPAMS,
                         x_axes=[NUM_DATA, NUM_MEDOIDS],
                         y_axes=[SAMPLE_COMPLEXITY, RUNTIME],
                         is_logspace_y=True
                         )
