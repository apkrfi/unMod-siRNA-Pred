datapath = "raw_data/All_Dataset.csv"
representation_dir = "representations" #or single_window_representations
outpath = "LOGO_outfolder" #or LOGO_outfolder

"""
datapath
=========================
path to raw data/dataset

-full dataset: raw_data

representation_dir
=========================
path to representation data/ feature(X) space

- all sequence representation: representations
- z-curve representations: single_window_representations

outpath
=========================
path to outputs/ metrics for various tests

- LOGO: LOGO_outfolder
- single window window: single_window_outfolder
- sliding window: sliding_window_outfolder
- fixed size LOGO, Holdout test (on other data), databound test, K-fold test: other_CV_tests

"""
