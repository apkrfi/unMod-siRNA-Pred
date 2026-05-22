import pandas as pd
import os
import numpy as np
from basic import *
from chaos import *
from numerical_mappings import *
from physico_chemical import *
from CN import *
from config import *

"""
Generate single window z-curve representations
"""
representations = ["zcurve"]
data = list(pd.read_csv(datapath)["siRNA_seq"])  # change directory in config file

for i in range(0, len(data[0])):
    window_data = [seq[i] for seq in data]
    for r in representations:
        df = globals()[r](window_data)  # calls functions by names in r
        df.to_csv(
            os.path.join(representation_dir, r + "_" + str(i) + ".csv")
        )  # change directory in config
