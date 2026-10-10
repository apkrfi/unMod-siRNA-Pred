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
"""
This is the main model and cross-validation section.
"""

 
def pairwise_distance_matrix(seqs, mode="global"):
    """Distance = 1 - (matching positions in best alignment / length of longer seq)."""
    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.gap_score = 0  # score == number of identical aligned positions
 
    n = len(seqs)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            matches = aligner.score(seqs[i], seqs[j])
            d = 1.0 - matches / max(len(seqs[i]), len(seqs[j]))
            dist[i, j] = dist[j, i] = d
    return dist
 
 
def cluster_sequences(df, seq_col="siRNA_seq", method="a"):
    """
    Hierarchically cluster sequences with Bio.Cluster.treecluster and add a 'Cluster' column.
 
    n_clusters : cut the tree into exactly this many clusters
    threshold  : OR cut the tree at this distance (0-1, e.g. 0.2 = ~80% identity)
    method     : 's' single, 'm' complete (maximum), 'a' average (UPGMA)
    """
 
    seqs = df[seq_col].astype(str).str.upper().tolist()
    if len(seqs) < 2:
        out = df.copy()
        out["Cluster"] = 0
        return out
 
    dist = pairwise_distance_matrix(seqs)
    print(np.mean(dist))
    threshold=round(np.mean(dist),1)# 60% similarity 
    tree = treecluster(None, distancematrix=dist, method=method)
 
    # each merge at or below the threshold reduces the cluster count by one
    merges = sum(node.distance <= threshold for node in (tree[k] for k in range(len(tree))))
    n_clusters = len(seqs) - merges
    print(n_clusters)
 
    labels = tree.cut(n_clusters)
 
    # renumber so clusters are 1, 2, 3... in order of first appearance
    mapping = {old: new for new, old in enumerate(dict.fromkeys(labels), start=1)}
 
    out = df.copy()
    out["Cluster"] = ['cluster_'+str(mapping[l]) for l in labels]
    print(out.head(5))
    out.to_csv('clustered.csv', index=False)
    return out


class Dataset:
    def __init__(self, X, Y, train_i, test_i):
        self.train_X = X.loc[train_i]
        self.train_Y = Y.loc[train_i]
        self.test_X = X.loc[test_i]
        self.test_Y = Y.loc[test_i]


class Run_Model:
    def __init__(self,representations, outfile_name, huesken=True, label_col='', keep_preds=False):
        print("Huesken only? ", huesken)
        self.outfile_name = outfile_name
        self.keep_preds=keep_preds
        self.predictions={"representations": representations}
        self.metrics = {"representations": representations}
        self.data = pd.read_csv(datapath)
        if huesken == True:
            huesken_index = self.data.index[self.data["sources"] == "Huesken"].tolist()
            self.data = self.data.loc[huesken_index]
        self.data = self.data.rename(columns={"Label": "label70"})
        self.data["label60"] = (self.data["efficacy"] > 0.6).astype(int)
        self.data["label80"] = (self.data["efficacy"] > 0.8).astype(int)
        self.data=cluster_sequences(self.data)
        self.X = pd.DataFrame()
        for rep in representations:  # in case u wanna combine multiple reps
            df = pd.read_csv(os.path.join(representation_dir, rep + ".csv")).iloc[:, 1:]
            df = df.add_prefix(f"{rep}__")
            self.X = pd.concat([self.X, df], axis=1)
            self.X = self.X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

            if(label_col==''):
                label_col='label70'
            self.Y = self.data[label_col]
            self.groups = self.data["B"]
            self.clusters = self.data['Cluster']
        if huesken == True:
            self.X = self.X.loc[huesken_index]#index always in order cus Huesken first half
        # print(self.X)

    def holdout_run(self):
        '''
        train: Huesken
        test: Remaining referred as independent set in paper
        '''
        model_names = ["SVC", "RF", "LGR", "DT", "XGB", "KNN", "SGD"]
        train_index = self.data.index[self.data["sources"] == "Huesken"].tolist()
        test_index = self.data.index[self.data["sources"] == "Other"].tolist()
        test_group = "Other"
        print("Train Length: ", len(train_index))
        print("Test Length: ", len(test_index))
        self.metrics[test_group] = {}
        dataset = Dataset(self.X, self.Y, train_index, test_index)
        for nm in model_names:
            model = Model(nm, dataset)
            model.create_model()
            model.run_model()
            self.metrics[test_group][nm] = model.metric_dictionary

        with open(os.path.join(outpath, self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp)

    def kfold_run(self):
        '''
        10-fold cross-validation
        '''
        widgets = [
            " [",
            progressbar.Timer(format="elapsed time: %(elapsed)s"),
            "] ",
            progressbar.Bar("#"),
            " (",
            progressbar.ETA(),
            ") ",
        ]
        bar = progressbar.ProgressBar(max_value=10, widgets=widgets).start()
        kf = KFold(n_splits=10, shuffle=True)
        model_names = ["SVC", "RF", "LGR", "DT", "XGB", "KNN", "SGD"]
        for i, (train_index, test_index) in enumerate(kf.split(self.X, self.Y)):
            test_group = "Fold_" + str(i)
            self.metrics[test_group] = {}
            dataset = Dataset(self.X, self.Y, train_index, test_index)
            for nm in model_names:
                model = Model(nm, dataset)
                model.create_model()
                model.run_model()
                self.metrics[test_group][nm] = model.metric_dictionary
            bar.update(i)
        # print(self.metrics)
        with open(os.path.join(outpath, self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp)
    def ensemble_run(self):
        '''
        Leave one group(target) out AND save predictions
        '''
        widgets = [
            " [",
            progressbar.Timer(format="elapsed time: %(elapsed)s"),
            "] ",
            progressbar.Bar("#"),
            " (",
            progressbar.ETA(),
            ") ",
        ]
        bar = progressbar.ProgressBar(max_value=30, widgets=widgets).start()
        logo = LeaveOneGroupOut()
        model_names = ["VOTE"]
        for i, (train_index, test_index) in enumerate(
            logo.split(self.X, self.Y, self.groups)
        ):
            test_group = np.unique(np.asarray(self.groups)[test_index])[0]
            self.metrics[test_group] = {}
            dataset = Dataset(self.X, self.Y, train_index, test_index)
            for nm in model_names:
                model = Model(nm, dataset)
                model.create_model()
                model.run_model()            
            self.metrics[test_group][nm] = model.metric_dictionary
            bar.update(i)
        # print(self.metrics)
        with open(os.path.join(outpath, self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp)
    def logo_run(self):
        '''
        Leave one group(target) out AND save predictions
        '''
        widgets = [
            " [",
            progressbar.Timer(format="elapsed time: %(elapsed)s"),
            "] ",
            progressbar.Bar("#"),
            " (",
            progressbar.ETA(),
            ") ",
        ]
        bar = progressbar.ProgressBar(max_value=30, widgets=widgets).start()
        logo = LeaveOneGroupOut()
        model_names = ["SVC", "RF", "LGR", "DT", "XGB", "KNN", "SGD"]
        for i, (train_index, test_index) in enumerate(
            logo.split(self.X, self.Y, self.groups)
        ):
            test_group = np.unique(np.asarray(self.groups)[test_index])[0]
            self.metrics[test_group] = {}
            self.predictions[test_group]={}
            dataset = Dataset(self.X, self.Y, train_index, test_index)
            for nm in model_names:
                model = Model(nm, dataset)
                model.create_model()
                model.run_model()
                self.predictions[test_group][nm] = list(zip(model.pred_y.tolist(), test_index.tolist()))                
                self.metrics[test_group][nm] = model.metric_dictionary
                bar.update(i)
        # print(self.metrics)
        with open(os.path.join(outpath, self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp)
        if(self.keep_preds==True):
            with open(os.path.join(outpath, 'Predictions_'+self.outfile_name), "w") as fp:
                        json.dump(self.predictions, fp)

    def logo_fixed_size_run(self):
        '''
        leave one group out with fixed training set size
        '''
        widgets = [
            " [",
            progressbar.Timer(format="elapsed time: %(elapsed)s"),
            "] ",
            progressbar.Bar("#"),
            " (",
            progressbar.ETA(),
            ") ",
        ]
        bar = progressbar.ProgressBar(max_value=30, widgets=widgets).start()
        logo = LeaveOneGroupOut()
        model_names = ["SVC", "RF", "LGR", "DT", "XGB", "KNN", "SGD"]
        for i, (train_index, test_index) in enumerate(
            logo.split(self.X, self.Y, self.groups)
        ):
            train_index = np.random.choice(
                train_index, size=len(self.groups) - 150, replace=False
            )
            print("Train Length: ", len(train_index))
            print("Test Length: ", len(test_index))
            test_group = np.unique(np.asarray(self.groups)[test_index])[0]
            self.metrics[test_group] = {}
            dataset = Dataset(self.X, self.Y, train_index, test_index)
            for nm in model_names:
                model = Model(nm, dataset)
                model.create_model()
                model.run_model()
                self.metrics[test_group][nm] = model.metric_dictionary
            bar.update(i)
        # print(self.metrics)
        with open(os.path.join(outpath, self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp)


    def loco_run(self):
        '''
        Leave one cluster(similar sequence) out
        '''
        widgets = [
            " [",
            progressbar.Timer(format="elapsed time: %(elapsed)s"),
            "] ",
            progressbar.Bar("#"),
            " (",
            progressbar.ETA(),
            ") ",
        ]
        bar = progressbar.ProgressBar(max_value=50, widgets=widgets).start()
        logo = LeaveOneGroupOut()
        model_names = ["SVC", "RF", "LGR", "DT", "XGB", "KNN", "SGD"]
        for i, (train_index, test_index) in enumerate(
            logo.split(self.X, self.Y, self.clusters)
        ):
            test_group = np.unique(np.asarray(self.clusters)[test_index])[0]
            self.metrics[test_group] = {}
            dataset = Dataset(self.X, self.Y, train_index, test_index)
            for nm in model_names:
                model = Model(nm, dataset)
                model.create_model()
                model.run_model()
                self.metrics[test_group][nm] = model.metric_dictionary
            bar.update(i)
        # print(self.metrics)
        with open(os.path.join(outpath, self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp)

    def databound_run(self):
        '''
        train: Huesken
        test: Reamining data aka Independent set 
        folds: increasing training set size run 9 times
        '''
        model_names = ["SVC", "RF", "LGR", "DT", "XGB", "KNN", "SGD"]
        train_index = self.data.index[self.data["sources"] == "Huesken"].tolist()
        test_index = self.data.index[self.data["sources"] == "Other"].tolist()
        n_data = list(range(50, len(train_index), 50))
        widgets = [
            " [",
            progressbar.Timer(format="elapsed time: %(elapsed)s"),
            "] ",
            progressbar.Bar("#"),
            " (",
            progressbar.ETA(),
            ") ",
        ]
        bar = progressbar.ProgressBar(
            max_value=10 * len(n_data), widgets=widgets
        ).start()

        bar_status = 0
        for n in n_data:
            for rep in range(1, 10):
                train_index_subset = np.random.choice(
                    train_index, size=n, replace=False
                )
                # print('Train Length: ',len(train_index_subset))
                # print('Test Length: ',len(test_index))
                test_group = str(n) + "_" + str(rep)
                self.metrics[test_group] = {}
                dataset = Dataset(self.X, self.Y, train_index_subset, test_index)
                for nm in model_names:
                    model = Model(nm, dataset)
                    model.create_model()
                    model.run_model()
                    self.metrics[test_group][nm] = model.metric_dictionary
                bar.update(bar_status)
                bar_status += 1

        with open(os.path.join(outpath, self.outfile_name), "w") as fp:
            json.dump(self.metrics, fp)



def main():
    parser = argparse.ArgumentParser(description="Run siRNA Models")
    parser.add_argument(
        "--representations", "-r",
        nargs="+",
        default=["zcurve"],
        choices=[
            "zcurve", "one_hot", "real", "EIIP", "atomic_number",
            "classifical_chaos", "complex_network_features", "frequency_chaos",
            "GC_count", "global_descriptor", "integer", "kmer_counts"
        ],
        metavar="REP",
        help="One or more representations to test (default: zcurve)"
    )
    parser.add_argument(
        "--cv", "-c",
        default="logo",
        choices=["logo", "logo_fixed_size", "holdout", "databound", "kfold"
                , 'loco'],
        help="Cross-validation method to use (default: holdout)"
    )
    parser.add_argument(
        "--huesken",
        action="store_true",
        help="Enable huesken mode (use for logo/logo_fixed_size/kfold; omit for holdout/databound)"
    )
    parser.add_argument(
        "--output-suffix",
        default="_logo.json",
        help="Suffix for output JSON filenames (default: _holdout.json)"
    )
    parser.add_argument(
            "--label_col",
            default="",
                    choices=["label70","label60","label80"],
            help="threshold"
        )
    args = parser.parse_args()
    print(args)

    start = time.perf_counter()

    for rep in args.representations:
        print("Starting for: " + rep)
        rm = Run_Model([rep], rep + args.label_col+args.output_suffix, huesken=args.huesken, label_col=args.label_col)
        getattr(rm, f"{args.cv}_run")()

    end = time.perf_counter()
    print(f"Total time: {end - start:.2f} seconds")# added some more details to function so speed may change

if __name__ == "__main__":
    main()
    #usage python run_models.py -r zcurve --output-suffix "_logo.json" -c logo --huesken
