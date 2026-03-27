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

def one_hot(sequences, max_length=None):
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
    if not sequences:
        return pd.DataFrame()

    sequences = [str(seq).upper() for seq in sequences]

    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    encoded_records = []

    for seq in sequences:
        # Truncate if longer than max_length
        seq = seq[:max_length]

        A, C, T, G = [], [], [], []

        for nucle in seq:
            A.append(1 if nucle == "A" else 0)
            C.append(1 if nucle == "C" else 0)
            T.append(1 if nucle in ("T", "U") else 0)
            G.append(1 if nucle == "G" else 0)

        concat = A + C + T + G

        # Pad to max_length for each nucleotide block
        padding = (max_length - len(seq)) * 4
        mapping = np.pad(concat, (0, padding), mode="constant")

        encoded_records.append(mapping)

    # Create column names
    columns = [f"pos{i+1}" for i in range(max_length * 4)]

    return pd.DataFrame(
        encoded_records,
        columns=columns[:encoded_records[0].shape[0]],
        index=[f"seq{i+1}" for i in range(len(sequences))]
    )


    
