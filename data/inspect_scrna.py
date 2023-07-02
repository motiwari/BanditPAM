import pandas as pd

# Define the directory where the data batches are stored
dataset = "scrna_reformat.csv"
df = pd.read_csv(dataset)

print(df.shape)

df = df[:100]

# Write to CSV
df.to_csv("smaller_scrna.csv", header=False, index=False)
