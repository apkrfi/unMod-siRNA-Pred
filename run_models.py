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
import argparse

"""
This is the main model and cross-validation section.
"""

class Dataset:
    def __init__(self, X, Y, train_i, test_i):
        self.train_X = X.loc[train_i]
        self.train_Y = Y.loc[train_i]
        self.test_X = X.loc[test_i]
        self.test_Y = Y.loc[test_i]


class Run_Model:
    def __init__(self, representations, outfile_name, huesken=True):
        print("Huesken only? ", huesken)
        self.outfile_name = outfile_name
        self.metrics = {"representations": representations}
        self.data = pd.read_csv(datapath)
        if huesken == True:
            huesken_index = self.data.index[self.data["sources"] == "Huesken"].tolist()
            self.data = self.data.loc[huesken_index]
        self.X = pd.DataFrame()
        for rep in representations:#in case u wanna combine multiple reps
            self.X = pd.concat(
                [
                    self.X,
                    pd.read_csv(os.path.join(representation_dir, rep + ".csv")).iloc[
                        :, 1:
                    ],
                ],
                axis=1,
            )
            self.X = self.X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
            self.Y = self.data["Label"]
            self.groups = self.data["B"]
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

    def logo_run(self):
        '''
        Leave one group(target) out
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

    def logo_fixed_size(self):
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
        choices=["logo", "logo_fixed_size", "holdout", "databound", "kfold"],
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
    args = parser.parse_args()

    start = time.perf_counter()

    for rep in args.representations:
        print("Starting for: " + rep)
        rm = Run_Model([rep], rep + args.output_suffix, huesken=args.huesken)
        getattr(rm, f"{args.cv}_run")()

    end = time.perf_counter()
    print(f"Total time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
    #usage python run_models.py -r zcurve --output-suffix "_logo.json" -c logo --huesken
