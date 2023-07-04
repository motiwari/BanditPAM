import os
import pandas as pd
from pandas import read_csv

source = os.listdir("scrna")

for s in source:
    csv = pd.read_csv(os.path.join("scrna", s))
    new_index = int(s[-5])+5
    new_s = s[:-5] + str(new_index) + s[-4:]
    dest = os.path.join("scrna", new_s)
    print(dest)
    csv.to_csv(dest, index=False)