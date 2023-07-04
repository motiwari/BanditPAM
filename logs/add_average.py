import pandas as pd
import glob
import os

# Get the current directory
current_directory = os.getcwd()

# # Find all CSV files in current directory and its subdirectories
# for file in glob.glob(current_directory + '/**/*.csv', recursive=True):
#     if "k_cluster" in file: continue
#     csv = pd.read_csv(file)
#     csv["average_complexity_with_caching"] = csv[
#         "total_complexity_with_caching"] / (csv["number_of_swaps"] + 1)
#     csv["average_runtime"] = csv["total_runtime"] / (csv["number_of_swaps"] + 1)
#     csv.to_csv(file, index=False)

# for file in glob.glob(
#     current_directory + "/**/*Original*.csv", recursive=True
# ):
#     if "k_cluster" in file:
#         continue
#     print(file)
#     csv = pd.read_csv(file)
#     new_num_swaps = csv["number_of_swaps"] / 2
#     print(csv["number_of_swaps"])
#     # csv["number_of_swaps"] = new_num_swaps
#     # csv["average_complexity_with_caching"] = csv[
#     #     "total_complexity_with_caching"
#     # ] / (new_num_swaps + 1)
#     # csv["average_runtime"] = csv["total_runtime"] / (new_num_swaps + 1)
#     # csv.to_csv(file, index=False)

files = os.listdir("scrna")
