import pandas as pd
from collections import Counter
import itertools
import numpy as np

def kmer_counts(sequences, k=3):
    """
    Returns K-mer counts for the given sequences.

    Parameters
    ----------
    sequences : list[str]
        A list of RNA sequences.
        
    k : int
        k value for k-mers

    Returns
    -------
    pandas.DataFrame
        DF with counts for all possible k-mers for the input sequence.

    """
    alphabet = "ACGT"
    all_kmers = [''.join(p) for p in itertools.product(alphabet, repeat=k)]
    
    records = []
    for seq in sequences:
        counts = Counter([seq[i:i+k] for i in range(len(seq)-k+1)])
        row = {kmer: counts.get(kmer, 0) for kmer in all_kmers}
        records.append(row)
    
    return pd.DataFrame(records, index=[f"seq{i+1}" for i in range(len(sequences))])

def one_hot(sequences):
    """
    Returns One Hot Encodings for the given sequences.

    Parameters
    ----------
    sequences : list[str]
        A list of RNA sequences.

    Returns
    -------
    pandas.DataFrame
        DF with one hot encodings of the input sequence.

    """
    alphabet = "ACGT"
    mapping = {char: i for i, char in enumerate(alphabet)}

    encoded_records = []
    for seq in sequences:
        mat = np.zeros((len(seq), len(alphabet)), dtype=int)
        for pos, char in enumerate(seq):
            if char in mapping:  
                mat[pos, mapping[char]] = 1
        encoded_records.append(mat.flatten())
    columns = []
    for pos in range(max(len(s) for s in sequences)):
        for char in alphabet:
            columns.append(f"pos{pos}_{char}")
    
    return pd.DataFrame(encoded_records, columns=columns[:encoded_records[0].shape[0]],
                        index=[f"seq{i+1}" for i in range(len(sequences))])


    
