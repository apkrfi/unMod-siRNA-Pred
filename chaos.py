import numpy as np
import pandas as pd
from collections import Counter


def frequency_chaos(sequences, k=3, max_length=None):
    """
    Returns Frequency Chaos Game Representations for sequences.

    Parameters
    ----------
    sequences : list[str]
        A list of RNA sequences.
    k : int
        k value for k-mers
    max_length: int
                maximum length of sequences

    Returns
    -------
    pandas.DataFrame
        DF with FCGRs for all sequences.

    """
    features = []
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    for seq in sequences:
        seq = seq.upper()

        # Count k-mers across whole sequence
        total_windows = max(len(seq) - k + 1, 1)
        kmer_counts = Counter([seq[i : i + k] for i in range(len(seq) - k + 1)])

        # Mapping = normalized frequency for each sliding window
        mapping = [
            kmer_counts[seq[i : i + k]] / total_windows for i in range(len(seq) - k + 1)
        ]

        # Pad
        padding = (max_length - k + 1) - len(mapping)
        mapping = np.pad(mapping, (0, max(0, padding)), "constant")

        features.append(mapping)

    return pd.DataFrame(features, index=[f"seq{i + 1}" for i in range(len(sequences))])


def classifical_chaos(sequences, max_length=None):
    """
    Returns Chaos Game Representations for sequences.

    Parameters
    ----------
    sequences : list[str]
        A list of RNA sequences.
    max_length : int
        Maximum length of sequences.

    Returns
    -------
    pandas.DataFrame
        DF with classical chaos representations.

    """
    features = []
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    for seq in sequences:
        seq = seq.upper()

        # Map nucleotides to chaos values
        Sx, Sy = [], []
        for nucle in seq:
            if nucle == "A":
                Sx.append(1)
                Sy.append(1)
            elif nucle == "C":
                Sx.append(-1)
                Sy.append(-1)
            elif nucle in ("T", "U"):
                Sx.append(-1)
                Sy.append(1)
            else:  # G or other
                Sx.append(1)
                Sy.append(-1)

        # Build CGR trajectory
        CGR_x, CGR_y = [], []
        for i in range(len(Sx)):
            if i == 0:
                CGR_x.append(0.5 * Sx[i])
                CGR_y.append(0.5 * Sy[i])
            else:
                CGR_x.append(0.5 * Sx[i] + 0.5 * CGR_x[i - 1])
                CGR_y.append(0.5 * Sy[i] + 0.5 * CGR_y[i - 1])

        concat = CGR_x + CGR_y

        # Pad
        padding = (max_length - len(Sx)) * 2
        mapping = np.pad(concat, (0, max(0, padding)), "constant")

        features.append(mapping)

    return pd.DataFrame(features, index=[f"seq{i + 1}" for i in range(len(sequences))])
