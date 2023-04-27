import pandas as pd

# Load the CSV file into a Pandas dataframe
df = pd.read_csv("scrna_reformat.csv", header=None)

print(df.describe())

# Get the first 10,000 rows
# df_10k = df.head(100)

# print(df_10k.describe())

# Save the first 10,000 rows to a new CSV file
# df_10k.to_csv("mnist_10k.csv", index=False)

