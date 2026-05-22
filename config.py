datapath = "raw_data/All_Dataset.csv"
representation_dir = "representations"
outpath = "other_CV_tests"

"""
datapath
=========================
path to raw data/dataset

-full dataset: raw_data

representation_dir
=========================
path to representation data/ feature(X) space

- all sequence representation: representations
- z-curve representations: sliding_window_representations

outpath
=========================
path to outputs/ metrics for various tests

- LOGO: LOGO_outfolder
- sliding window: sliding_outfolder
- fixed size LOGO, Holdout test (on other data), databound test: more_tests
"""
