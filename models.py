import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupShuffleSplit, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore') 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
import progressbar

class Model:

    def __init__(self,model_name,dataset):
        self.name=model_name
        self.dataset=dataset
        self.clf=None
        self.pred_y=[]
        self.scores_y=[]
        self.metric_dictionary={}

    def create_model(self):
        """ Create sklearn models """ 

        if(self.name=='RF'):
            self.clf = RandomForestClassifier(n_estimators=1000,max_depth=2, random_state=42)#n_estimators-->reduced
        elif(self.name=='MLP'):
            self.clf = MLPClassifier(solver='adam', max_iter=5000,activation='logistic',alpha=1e-5,hidden_layer_sizes=(62, 12), random_state=42,learning_rate='adaptive')#https://doi.org/10.1016/j.ygeno.2024.110815
        elif(self.name=='XGB'):
            self.clf=XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
        elif(self.name == 'SVC'):
            self.clf = SVC(kernel='rbf', probability=True, random_state=42,max_iter=5000) 
        elif(self.name == 'LGR'):
            self.clf = LogisticRegression(random_state=42)
        elif(self.name == 'DT'):
            self.clf = tree.DecisionTreeClassifier()
        elif(self.name == 'KNN'):
            self.clf = KNeighborsClassifier(n_neighbors=7)
        elif(self.name == 'SGD'):
            self.clf = SGDClassifier(loss="log_loss", max_iter=5)
    
    def get_metrics(self):
        """ Generate classification metrics """ 

        fpr,tpr,thresholds = metrics.roc_curve(self.dataset.test_Y, self.scores_y)
        self.metric_dictionary['roc_auc'] = metrics.auc(fpr, tpr)
        self.metric_dictionary['pr_auc'] = metrics.average_precision_score(self.dataset.test_Y, self.scores_y)
        self.metric_dictionary['precision'],self.metric_dictionary['recall'],self.metric_dictionary['fscore'],_=precision_recall_fscore_support(self.dataset.test_Y, self.pred_y, average='binary')
        
        

    def run_model(self):
        """ runs model for instance's dataset and model type """

        self.clf.fit(self.dataset.train_X, self.dataset.train_Y)
        self.pred_y=self.clf.predict(self.dataset.test_X)
        self.scores_y=self.clf.predict_proba(self.dataset.test_X)[:, 1]
        self.get_metrics()



        