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
- Z-curve representations
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
### 1. Create environment
### 2. Install packages
### 3. Generate Representations
### 4. Generate Single Window Representations
### 5. Run Tests

a. Leave One Group Out

b. Leave One Group Out Fixed Size

c. Holdout

d. Databound

e. K-fold

### 6. Run Sliding Window Tests



