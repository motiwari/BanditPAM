from typing import List
import os
from matplotlib import pyplot as plt
import glob
import numpy as np
import pandas as pd

from constants import (
    MNIST,
    # algorithms
    BANDITPAM_ORIGINAL_NO_CACHING,
    ALL_BANDITPAMS,
    # experiment settings
    NUM_DATA,
    RUNTIME,
    SAMPLE_COMPLEXITY,
    LOSS,
    # utils
    ALG_TO_COLOR,
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
            log_dir_name = (
                "scaling_with_n_cluster"
                if x_axis == NUM_DATA
                else "scaling_with_k_cluster"
            )
            log_dir = os.path.join(parent_dir, "logs", log_dir_name)
        else:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(root_dir, "logs", dir_name)

        for dataset in datasets:
            csv_files = glob.glob(os.path.join(log_dir, f"*{dataset}*"))

            # list available settings by parsing the log file names
            # for example, "settings" returns ["k5", "k10"] if the experiment used "k = 5" and "k = 10" settings
            settings = list(set([file.split("_")[-2] for file in csv_files]))

            for setting in settings:
                for y_axis in y_axes:
                    banditpam_original_losses = None
                    for algorithm in ALL_BANDITPAMS:
                        algorithm_files = glob.glob(
                            os.path.join(
                                log_dir, f"*{algorithm}*{dataset}*" f"{setting}*idx*"
                            )
                        )
                        algorithm_dfs = [pd.read_csv(file) for file in algorithm_files]
                        data = pd.concat(algorithm_dfs)

                        # Calculate the mean of each row across the files
                        data_mean = data.groupby(data.index).mean()
                        data_std = data.groupby(data.index).std() / np.sqrt(len(data))

                        # Set x axis
                        xlabel = (
                            "Dataset size ($n$)"
                            if x_axis == NUM_DATA
                            else "Number of medoids ($k$)"
                        )
                        if is_logspace_x:
                            xlabel = f"ln({xlabel})"
                        x = data[x_axis].tolist()

                        # Set y axis
                        if y_axis is LOSS:
                            # Plot the loss divided by that of Original BanaditPam without Caching
                            y = np.array(data_mean[y_axis].tolist())
                            if algorithm == BANDITPAM_ORIGINAL_NO_CACHING:
                                # The first algorithm is BANDITPAM_ORIGINAL_NO_CACHING
                                banditpam_original_losses = y.copy()
                            y /= banditpam_original_losses
                            error = data_std[y_axis].tolist()
                            ylabel = (
                                "Final Loss Normalized to BanditPAM ($L/L_{BanditPAM}$)"
                            )
                        elif y_axis is SAMPLE_COMPLEXITY:
                            y = data_mean["total_complexity_with_caching"].tolist()
                            error = data_std["total_complexity_with_caching"].tolist()
                            ylabel = "Sample Complexity"
                        else:
                            y = data_mean["total_runtime"].tolist()
                            error = data_std["total_runtime"].tolist()
                            ylabel = "Wall-clock Runtime"

                        if is_logspace_y:
                            ylabel = f"ln({ylabel})"

                        # Sort the (x, y) pairs by the ascending order of x
                        x, y = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))

                        # Apply log to x, y and error
                        if is_logspace_x:
                            x = np.log(x)

                        if is_logspace_y:
                            y = np.log(y)
                            error = np.log(error) / y / np.sqrt(len(algorithm_files))

                        plt.scatter(
                            x, y, color=ALG_TO_COLOR[algorithm], label=algorithm,
                        )
                        plt.plot(x, y, color=ALG_TO_COLOR[algorithm])

                        if include_error_bar:
                            plt.errorbar(
                                x, y, yerr=np.abs(error), fmt=".", color="black"
                            )

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
                            xtitle = "$n$"
                        title = f"{ytitle} vs. {xtitle} {translate_experiment_setting(dataset, setting)}"
                        plt.title(title)
                        plt.xlabel(xlabel)
                        plt.ylabel(ylabel)

                    plt.show()


if __name__ == "__main__":
    create_scaling_plots(
        datasets=[MNIST],
        algorithms=[ALL_BANDITPAMS],
        x_axes=[NUM_DATA],
        y_axes=[SAMPLE_COMPLEXITY, RUNTIME],
        is_logspace_y=False,
        dir_name="scaling_with_n",
        include_error_bar=True,
    )
    create_scaling_plots(
        datasets=[MNIST],
        algorithms=[ALL_BANDITPAMS],
        x_axes=[NUM_DATA],
        y_axes=[LOSS],
        is_logspace_y=False,
        dir_name="scaling_with_n",
        include_error_bar=False,
    )
