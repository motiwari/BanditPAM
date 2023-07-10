import os
import pandas as pd
from pandas import read_csv

source = os.listdir("scrna_backup")

for s in source:
    csv = pd.read_csv(os.path.join("scrna_backup", s))
    new_index = int(s[-5]) + 5
    new_s = s[:-5] + str(new_index) + s[-4:]
    dest = os.path.join("scrna_backup", new_s)
    print(dest)
    csv.to_csv(dest, index=False)
