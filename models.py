import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    GroupShuffleSplit,
    train_test_split,
    GridSearchCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")
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
    def __init__(self, model_name, dataset):
        self.name = model_name
        self.dataset = dataset
        self.clf = None
        self.pred_y = []
        self.scores_y = []
        self.metric_dictionary = {}

    def create_model(self):
        """Create sklearn models with default parameters"""

        if self.name == "RF":
            self.clf = RandomForestClassifier(random_state=42)  # n_estimators-->reduced
        elif self.name == "MLP":
            self.clf = MLPClassifier(
                random_state=42
            )  # https://doi.org/10.1016/j.ygeno.2024.110815
        elif self.name == "XGB":
            self.clf = XGBClassifier(random_state=42, objective="binary:logistic")
        elif self.name == "SVC":
            self.clf = SVC(probability=True, random_state=42)
        elif self.name == "LGR":
            self.clf = LogisticRegression(random_state=42)
        elif self.name == "DT":
            self.clf = tree.DecisionTreeClassifier(random_state=42)
        elif self.name == "KNN":
            self.clf = KNeighborsClassifier()
        elif self.name == "SGD":
            self.clf = SGDClassifier(random_state=42, loss="log_loss")

    def get_metrics(self):
        """Generate classification metrics"""

        fpr, tpr, thresholds = metrics.roc_curve(self.dataset.test_Y, self.scores_y)
        self.metric_dictionary["roc_auc"] = metrics.auc(fpr, tpr)
        self.metric_dictionary["pr_auc"] = metrics.average_precision_score(
            self.dataset.test_Y, self.scores_y
        )
        (
            self.metric_dictionary["precision"],
            self.metric_dictionary["recall"],
            self.metric_dictionary["fscore"],
            _,
        ) = precision_recall_fscore_support(
            self.dataset.test_Y, self.pred_y, average="binary"
        )

    def run_model(self):
        """runs model for instance's dataset and model type"""

        self.clf.fit(self.dataset.train_X, self.dataset.train_Y)
        self.pred_y = self.clf.predict(self.dataset.test_X)
        self.scores_y = self.clf.predict_proba(self.dataset.test_X)[:, 1]
        self.get_metrics()
