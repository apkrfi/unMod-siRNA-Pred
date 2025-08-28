import pandas as pd
import os
import numpy as np
from basic import *
from chaos import *
from foundation_models import *
from numerical_mappings import *
from physico_chemical import *
from CN import *
from config import *

#representations=['get_ctd','one_hot','kmer_counts','GC_count','zcurve','integer','atomic_number','EIIP','complex_network_features','classifical_chaos','frequency_chaos','real']
representations=['RNA_BERT','RinaLMO','RNA_Ernie']
data=list(pd.read_csv(datapath)['siRNA_seq'])
for r in representations:
    df=globals()[r](data)
    df.to_csv(os.path.join(representation_dir,r+'.csv'))

