import pandas as pd
from glob import glob
from scripts.constants import (
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_NO_CACHING,
    BANDITPAM_VA_CACHING,
)

# for algorithm in [
#     BANDITPAM_ORIGINAL_NO_CACHING,
#     BANDITPAM_ORIGINAL_CACHING,
#     BANDITPAM_VA_NO_CACHING,
#     BANDITPAM_VA_CACHING,
# ]:
#     print(algorithm)
#     csv_files = glob(f"scrna_backup/*{algorithm}*")
#     algorithm_dfs = [pd.read_csv(file) for file in csv_files]
#     data = pd.concat(algorithm_dfs)
#
#     # Calculate the mean of each row across the files
#     data_mean = data.groupby(data.index).mean()
#     print(data_mean["number_of_swaps"].mean())


for i in [0, 1, 2, 4, 5, 6]:
    print("I: ", i)
    for algorithm in [
        BANDITPAM_ORIGINAL_NO_CACHING,
        BANDITPAM_ORIGINAL_CACHING,
        BANDITPAM_VA_NO_CACHING,
        BANDITPAM_VA_CACHING,
    ]:
        print(algorithm)
        csv_file = glob(f"scrna_backup/*{algorithm}*idx{i}*")[0]
        csv = pd.read_csv(csv_file)
        print(csv["number_of_swaps"])
    print("--------------------")
    print("--------------------")
    print("--------------------")
