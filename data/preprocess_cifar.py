from email import header

import pandas as pd
import os

path = "cifar10.csv"
csv = pd.read_csv(path)
first_column_label = csv.columns[0]
csv = csv.drop(first_column_label, axis=1)
csv.to_csv(path, header=False, index=False)