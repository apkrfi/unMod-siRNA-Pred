# Project Description
This project aims to design a fast and accurate siRNA efficacy prediction pipeline based on the experiments discussed in [Benchmarking siRNA Prediction: The Role of Representation and Validation Strategies](https://www.biorxiv.org/content/10.64898/2026.05.12.724560v1.full.pdf)

The work explore various sequence representation techniques and cross validation strategies for optimizing siRNA acticity prediction.

# Datasets

Datasets were extracted from La Rosa's experiments. Our version includes a cleaned version with siRNA/mRNA names and sequences, efficacy values, corresponding Labels (efficacy>0.7: 1/High, efficacy<0.7: 0/Low) and sources i.e. Huesken/Others (refer paper). 

# Representations 

We have used the following representations which can be found in the corresponding scripts =>

### Types

##### Basic
Found in basic.py we extract
- k-mer counts
- one-hot encodings

##### Physico-chemical features
Found in physico_chemical.py we extract
- GC content
- CTD (Composition-Transition-Distribution) features

##### Numeric
Found in numerical_mappings.py we extract
- zcurve representations
- Integer representations
- EIIP (Electron-Ion Interaction Pseudopotentials)
- Atomic Number based representations
- Real representations

##### Chaos Game Representations
Found in chaos.py and originally proposed for DNA sequences by Jeffrey in 1990, we extract
- Classical Chaos Game Representations
- Frequency Chaos Game Representations

##### Complex Networks (BASiNET)
Found in CN.py we extract
- Graph features based on BASiNET's complex network method

Details of these representations can be found in the paper supplementary section.

### Representation Data
Representations were gereated for:

- Full dataset found in => \\representations
- Each nucleotide (0th-21st) found in ==> \\single_window_representations

# Directory Structure
```
unmod-siRNA-Pred/
├── LOGO_outfolder/                               <-------- Leave One Group/Target Out outputs
├── other_CV_tests/                               <-------- Holdout, K-fold, Fixed-training size, Data bound tests
├── raw_data/                                     <-------- raw dataset
├── representations/                              <-------- all sequence representations
├── single_window_representations/                <-------- single nucleotide representations using z-curve
├── sliding_window_outfolder/                     <-------- sliding Window outputs
├── CN.py                                         <-------- Complex Network representation
├── basic.py                                      <-------- K-mer, One-hot encodings
├── chaos.py                                      <-------- Chaos Game Theory representations
├── numerical_mappings.py                         <-------- z-curve, integerer, real, atomic, EIIP representations
├── physico_chemical.py                           <-------- GC counts, Global sequence descriptors
├── get_representations.py                        <-------- generates representations
├── get_single_window_representations.py          <-------- generates single nucelotide representations
├── models.py                                     <-------- model declarations
├── run_models.py                                 <-------- Training/Evaluation
├── run_sliding_window.py                         <-------- sliding window experiment
├── config.py                                     <-------- some global declarations
├── Visualize Results.ipynb                       <-------- Notebook to visualize results
├── pyproject.toml                                <-------- project requirements
├── uv.lock                                       <-------- frozen environment

```

# Output

We have conducted several experiments on our datasets.

### Leave One Target Out Test

- Dataset used: Huesken
- Strategy: Leave one target gene out every round for testing while training on the rest
- Purpose: To look at individual target performance and get an unbiased cross-validation score

### Leave One Target Out (Fixed Size)

- Dataset used: Huesken
- Strategy: Leave one target gene out every round for testing while training on the rest with FIXED training size
- Purpose: To remove training size bias from LOGO

### Sliding Window (LOGO)

- Dataset used: Huesken
- Strategy: LOGO but for various windows over siRNA sequences
- Purpose: To look at the importance of different lengths and regions across siRNAs

### Test on External Dataset (Holdout)

- Dataset used: Huesken and Others
- Strategy: train on Huesken Test on Others
- Purpose: To see how the model generalizes

### K-fold Test

- Dataset used: Huesken
- Strategy: Perform a K-fold test on the dataset
- Purpose:  To see how the performance differs from LOTO

### Test for Databound Model

- Dataset used: Huesken and Others
- Strategy: train on Huesken Test on Others with increasing training size from 50-2500 (step 50)
- Purpose: To see how the training size influences model performance

# Usage
### 1. Clone Repo
```
git clone https://github.com/apkrfi/unMod-siRNA-Pred.git
cd unMod-siRNA-Pred
```
### 2. Create environment
```
conda create -n siRNA_unmod
conda activate siRNA_unmod
```
### 3. Install packages
Download uv using [uv installation steps]([https://example.com](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer))
```
uv sync
```
### 4. Generate Representations
```
python get_representations.py --datapath raw_data/All_Dataset.csv --representation_dir representations
```
### 5. Generate Single Window Representations
```
python get_single_window_representations.py 
```
### 6. Run Tests

a. Leave One Group Out
```
python run_models.py -r zcurve --output-suffix "_logo.json" -c logo --huesken
```

b. Leave One Group Out Fixed Size
```
python run_models.py -r zcurve --output-suffix "_logo_fixed_size_.json" -c logo_fixed_size --huesken
```

c. Holdout
```
python run_models.py -r zcurve --output-suffix "_holdout.json" -c holdout 
```

d. Databound
```
python run_models.py -r zcurve --output-suffix "_databound.json" -c databound 
```

e. K-fold
```
python run_models.py -r zcurve --output-suffix "_kfold.json" -c kfold --huesken
```

### 6. Run Sliding Window Tests
```
python run_sliding_window.py 
```



