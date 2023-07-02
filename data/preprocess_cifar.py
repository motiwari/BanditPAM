import pickle
import pandas as pd


# Function to load a single batch file
def load_cifar_batch(file):
    with open(file, "rb") as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


# Define the directory where the data batches are stored
directory = "cifar-10-batches-py/"

# List of data batch files to load
batches = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
]

dataframes = []
for batch in batches:
    batch_data = load_cifar_batch(directory + batch)
    df = pd.DataFrame(batch_data[b"data"])
    df["label"] = batch_data[b"labels"]
    dataframes.append(df)

# Combine all batch dataframes
df = pd.concat(dataframes)
df /= 255.0

# Write to CSV
df.to_csv("cifar10.csv", header=False, index=False)

print(df.shape)
print(df.describe())
