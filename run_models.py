import pandas as pd
import numpy as np
import os
from config import *
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupShuffleSplit, train_test_split, GridSearchCV
from models import *
import json
import time

class Dataset:
    def __init__(self, X,Y, train_i, test_i):
        self.train_X=X.loc[train_i]
        self.train_Y=Y.loc[train_i]
        self.test_X=X.loc[test_i]
        self.test_Y=Y.loc[test_i]

class Run_Model:
    def __init__(self, representations, outfile_name, huesken=True):
        print('Huesken only? ',huesken)
        self.outfile_name=outfile_name
        self.metrics={'representations':representations}
        self.data=pd.read_csv(datapath)
        if(huesken==True):
            huesken_index=self.data.index[self.data['sources']=='Huesken'].tolist()
            self.data=self.data.loc[huesken_index]
        self.X=pd.DataFrame()
        for rep in representations:
            self.X=pd.concat([self.X,pd.read_csv(os.path.join(representation_dir,rep+'.csv')).iloc[:,1:]],axis=1)
            self.X=self.X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
            self.Y=self.data['Label']
            self.groups=self.data['B']
        if(huesken==True):
            self.X=self.X.loc[huesken_index]
        #print(self.X)
    
    def holdout_run(self):
        model_names=['RF','LGR','DT','XGB','KNN', 'SGD']
        train_index=self.data.index[self.data['sources'] == 'Huesken'].tolist()
        test_index=self.data.index[self.data['sources'] == 'Other'].tolist()
        test_group='Other'
        print('Train Length: ',len(train_index))
        print('Test Length: ',len(test_index))
        self.metrics[test_group]={}
        dataset= Dataset(self.X,self.Y,train_index, test_index)
        for nm in model_names:
            model=Model(nm,dataset)
            model.create_model()
            model.run_model()
            self.metrics[test_group][nm]=model.metric_dictionary

        with open(os.path.join(outpath,self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp) 

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
        model_names=['SVC','RF','LGR','DT','XGB','KNN', 'SGD']
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

    def logo_run_fixed_size(self):
        widgets = [' [',
            progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
            '] ',
            progressbar.Bar('#'),' (',
            progressbar.ETA(), ') ',
            ]
        bar = progressbar.ProgressBar(max_value=30, 
                                widgets=widgets).start()
        logo = LeaveOneGroupOut()
        model_names=['RF','LGR','DT','XGB','KNN', 'SGD']
        for i, (train_index, test_index) in enumerate(logo.split(self.X, self.Y, self.groups)):
            train_index=np.random.choice(train_index, size=len(self.groups)-150, replace=False)
            print('Train Length: ',len(train_index))
            print('Test Length: ',len(test_index))
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


    def databound_test(self):
        model_names=['RF','LGR','DT','XGB','KNN', 'SGD']
        train_index=self.data.index[self.data['sources'] == 'Huesken'].tolist()
        test_index=self.data.index[self.data['sources'] == 'Other'].tolist()
        n_data=list(range(50, len(train_index), 50))
        widgets = [' [',
        progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
        '] ',
        progressbar.Bar('#'),' (',
        progressbar.ETA(), ') ',
        ]
        bar = progressbar.ProgressBar(max_value=10*len(n_data), 
                        widgets=widgets).start()
        
        bar_status=0
        for n in n_data:
            for rep in range(1,10):

                train_index_subset=np.random.choice(train_index, size=n, replace=False)
                # print('Train Length: ',len(train_index_subset))
                # print('Test Length: ',len(test_index))
                test_group=str(n)+'_'+str(rep)
                self.metrics[test_group]={}
                dataset= Dataset(self.X,self.Y,train_index_subset, test_index)
                for nm in model_names:
                    model=Model(nm,dataset)
                    model.create_model()
                    model.run_model()
                    self.metrics[test_group][nm]=model.metric_dictionary
                bar.update(bar_status)
                bar_status+=1

        with open(os.path.join(outpath,self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp) 


def main():
    start = time.perf_counter()
    representations=['zcurve','one_hot','real','RinaLMO','RNA_BERT','RNA_Ernie','EIIP','atomic_number','classifical_chaos','complex_network_features','frequency_chaos','GC_count','global_descriptor','integer','kmer_counts']
    for rep in representations:
        print('Starting for: '+rep)
        rm=Run_Model([rep],rep+'.json',huesken=True)
        rm.logo_run()
    end = time.perf_counter()
    print(f"Total time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()   