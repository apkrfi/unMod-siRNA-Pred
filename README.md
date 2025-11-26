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

##### Foundation Models (Large Language Models)
Found in foundation_models.py we extract
- RinaLMO embeddings
- RNA-BERT embeddings
- RNAernie embeddings


### Representation Data
Representations were gereated for:

- Full dataset found in => \\representations
- Each nucleotide (0th-21st) found in ==> \\sliding_window_representations

# Output

We have conducted several experiments on our datasets.

### Leave One Group Out Test

- Dataset used: Huesken
- Strategy: Leave one target gene out every round for testing while training on the rest
- Purpose: To look at individual target performance and get an unbiased cross-validation score

### Leave One Group Out (Fixed Size)

- Dataset used: Huesken
- Strategy: Leave one target gene out every round for testing while training on the rest with FIXED training size
- Purpose: To remove training size bias from LOGO

### Sliding Window (LOGO)

- Dataset used: Huesken
- Strategy: LOGO but for 1 nucleotide at a time
- Purpose: To look at the importance of individual nucleotide

### Test on External Dataset

- Dataset used: Huesken and Others
- Strategy: train on Huesken Test on Others
- Purpose: To see how the model generalizes

### Test for Databound Model

- Dataset used: Huesken and Others
- Strategy: train on Huesken Test on Others with increasing training size from 50-2500 (step 50)
- Purpose: To see how the training size influences model performance

# Visualizing Results

