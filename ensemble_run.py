import pandas as pd
import numpy as np
import os
from config import *
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    GroupShuffleSplit,
    train_test_split,
    GridSearchCV,
)
from models import *
import json
import time
from Bio import Align
from Bio.Cluster import treecluster
import argparse
import math
from run_models import *
from itertools import combinations

time_lst=[]
rep_names=[]
reps=['zcurve','one_hot','classifical_chaos']
rep_combos = [list(c) for r in range(1, len(reps) + 1) for c in combinations(reps, r)]
print(rep_combos)
for rep in rep_combos:
    # if('zcurve' not in rep or 'classifical_chaos' not in rep):
    #     continue
    start = time.perf_counter()
    name= rep if isinstance(rep, str) else "_".join(rep)
    rep_names.append(name)
    print("Starting for: " + name)
    rm = Run_Model(rep, name + '_ensemble.json', huesken=True, label_col='label70')
    getattr(rm, "ensemble_run")()
    end = time.perf_counter()
    print(f"Total time: {end - start:.2f} seconds")   #all ensemble runs
    time_lst.append(end - start)  
time_df=pd.DataFrame({'Rep':rep_names,'Time':time_lst})
time_df.to_csv(outpath+'/ensemble_time.csv')



