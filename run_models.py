import pandas as pd
import numpy as np
import os
from config import *
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupShuffleSplit, train_test_split, GridSearchCV
from models import *
import json
class Dataset:
    def __init__(self, X,Y, train_i, test_i):
        self.train_X=X.loc[train_i]
        self.train_Y=Y.loc[train_i]
        self.test_X=X.loc[test_i]
        self.test_Y=Y.loc[test_i]

class Run_Model:
    def __init__(self, representations, outfile_name):
        self.outfile_name=outfile_name
        self.metrics={'representations':representations}
        self.data=pd.read_csv(datapath)
        huesken_index=self.data.index[self.data['sources']=='Huesken'].tolist()
        self.data=self.data.loc[huesken_index]
        self.X=pd.DataFrame()
        for rep in representations:
            self.X=pd.concat([self.X,pd.read_csv(os.path.join(representation_dir,rep+'.csv')).iloc[:,1:]],axis=1)
            self.X=self.X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
            self.Y=self.data['Label']
            self.groups=self.data['B']
        self.X=self.X.loc[huesken_index]
        #print(self.X)
        
    def logo_run(self):
        widgets = [' [',
            progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
            '] ',
            progressbar.Bar('#'),' (',
            progressbar.ETA(), ') ',
            ]
        bar = progressbar.ProgressBar(max_value=30, 
                                widgets=widgets).start()
        logo = LeaveOneGroupOut()
        model_names=['RF','LGR','DT']
        for i, (train_index, test_index) in enumerate(logo.split(self.X, self.Y, self.groups)):
            test_group=np.unique(np.asarray(self.groups)[test_index])[0]
            self.metrics[test_group]={}
            dataset= Dataset(self.X,self.Y,train_index, test_index)
            for nm in model_names:
                model=Model(nm,dataset)
                model.create_model()
                model.run_model()
                self.metrics[test_group][nm]=model.metric_dictionary
            bar.update(i)
        #print(self.metrics)
        with open(os.path.join(outpath,self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp) 


def main():
    representations=['complex_network_features','RNA_Ernie']
    for rep in representations:
        print('Starting for: '+rep)
        rm=Run_Model([rep],rep+'.json')
        rm.logo_run()

if __name__ == "__main__":
    main()   