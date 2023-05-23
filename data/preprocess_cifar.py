import pickle
import pandas as pd
import os


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


dfs = []
for i in range(1, 6):
    file_path = os.path.join("cifar-10-batches-py", f'data_batch_{i}')
    data_batch = unpickle(file_path)

    features = data_batch[b'data']
    df = pd.DataFrame(features) / 255
    dfs += df,

merged_df = pd.concat(dfs, ignore_index=True)
first_column_label = merged_df.columns[0]
merged_df = merged_df.drop(first_column_label, axis=1)
merged_df.to_csv("cifar10_temp.csv", header=False, index=False)
