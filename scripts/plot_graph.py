from typing import List
import os
from matplotlib import pyplot as plt
import glob
from pandas import read_csv
import numpy as np
import pandas as pd

from constants import (
    # datasets
    MNIST,
    SCRNA,
    CIFAR,

    # algorithms
    BANDITPAM_ORIGINAL_NO_CACHING,
    ALL_BANDITPAMS,

    # experiment settings
    NUM_DATA,
    NUM_MEDOIDS,
    RUNTIME,
    SAMPLE_COMPLEXITY,
    LOSS,

    # utils
    ALG_TO_COLOR
)


def extract_algorithm_from_filename(filename):
    for algorithm in ALL_BANDITPAMS:
        if algorithm in filename:
            return algorithm


def translate_experiment_setting(dataset, setting):
    """
    Translate a setting into a human-readable format
    For example, "k5" becomes "Num medoids: 5"
    """
    if "k" in setting:
        return f"({dataset}, ${setting[0]}={setting[1:]}$)"
    elif "n" in setting:
        return f"({dataset}, ${setting[0]}={setting[1:]}$)"
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
        include_error_bar: bool = False,
        dir_name: str = None,
        is_multiple_experiments: bool = False,
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
            log_dir = os.path.join(parent_dir, "../experiments", "logs", log_dir_name)
        else:
            parent_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(parent_dir, "../experiments", "logs", dir_name)

        for dataset in datasets:
            csv_files = glob.glob(os.path.join(log_dir, f"*{dataset}*"))
            if is_multiple_experiments:
                csv_files = list(filter(lambda x: "idx" in x, csv_files))
            else:
                csv_files = list(filter(lambda x: "idx" not in x, csv_files))

            # list available settings by parsing the log file names
            # for example, "settings" returns ["k5", "k10"] if the experiment used "k = 5" and "k = 10" settings
            if is_multiple_experiments:
                settings = list(set([file.split("_")[-2] for file in csv_files]))
            else:
                settings = list(set([file.split("_")[-1][:-4] for file in csv_files]))

            for setting in settings:
                print(setting)
                for y_axis in y_axes:
                    for algorithm in ALL_BANDITPAMS:
                        if is_multiple_experiments:
                            algorithm_files = glob.glob(os.path.join(log_dir, f"*{algorithm}*{dataset}*"
                                                                              f"{setting}*idx0*"))
                        else:
                            algorithm_files = glob.glob(os.path.join(log_dir, f"*{algorithm}*{dataset}*{setting}*"))
                        algorithm_dfs = [pd.read_csv(file) for file in algorithm_files]
                        data = pd.concat(algorithm_dfs)

                        # Add a new column 'file_id' to distinguish each file
                        data['file_id'] = np.repeat(np.arange(len(algorithm_dfs)), [len(df) for df in algorithm_dfs])

                        # Calculate the mean of each row across the files
                        data = data.groupby(data.index).mean()
                        data_std = data.groupby(data.index).std()

                        # Drop the 'file_id' column as it's not needed anymore
                        data = data.drop(columns='file_id')

                        # Set x axis
                        xlabel = (
                            "Dataset size ($n$)" if x_axis == NUM_DATA else "Number of medoids ($k$)"
                        )
                        if is_logspace_x:
                            xlabel = f"ln({xlabel})"

                        x = data[x_axis].tolist()

                        # Set y axis
                        if y_axis is LOSS:
                            y = data[y_axis].tolist()
                            error = data_std[y_axis].tolist()
                            ylabel = "Final Loss Normalized to BanditPAM ($L/L_{BanditPAM}$)"
                        elif y_axis is SAMPLE_COMPLEXITY:
                            y = data["total_complexity_with_misc"].tolist()
                            error = data_std["total_complexity_with_misc"].tolist()
                            ylabel = "Sample Complexity"
                        else:
                            y = data["total_runtime"].tolist()
                            error = data_std["total_runtime"].tolist()
                            ylabel = "Wall-clock Runtime"

                        if is_logspace_y:
                            ylabel = f"ln({ylabel})"

                        x, y = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))  # sort

                        if is_logspace_x:
                            x = np.log(x)

                        if is_logspace_y:
                            y = np.log(y)
                            error = np.log(error) / y / np.sqrt(len(algorithm_files))

                        plt.scatter(
                            x,
                            y,
                            color=ALG_TO_COLOR[algorithm],
                            label=algorithm,
                        )
                        plt.plot(x, y, color=ALG_TO_COLOR[algorithm])

                        if include_error_bar:
                            plt.errorbar(x, y, yerr=np.abs(error), fmt=".", color="black")

                        # Sort the legend entries (labels and handles) by labels
                        handles, labels = plt.gca().get_legend_handles_labels()
                        labels, handles = zip(
                            *sorted(zip(labels, handles), key=lambda t: t[0])
                        )
                        plt.legend(handles, labels, loc="upper left")

                        if y_axis == LOSS:
                            ytitle = "$L/L_{BanditPAM}$"
                        else:
                            ytitle = ylabel
                        if x_axis == NUM_DATA:
                            xtitle = "$n$"
                        else:
                            xtitle = "$k$"
                        title = f"{ytitle} vs. {xtitle} {translate_experiment_setting(dataset, setting)}"
                        plt.title(title)
                        plt.xlabel(xlabel)
                        plt.ylabel(ylabel)

                    plt.show()


if __name__ == "__main__":
    create_scaling_plots(datasets=[CIFAR],
                         algorithms=[ALL_BANDITPAMS],
                         x_axes=[NUM_MEDOIDS],
                         y_axes=[RUNTIME],
                         dir_name="cifar_k_mac",
                         is_logspace_y=False,
                         is_multiple_experiments=True,
                         include_error_bar=True,
                         )
