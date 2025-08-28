import numpy as np
import pandas as pd

def zcurve(sequences, max_length=None):
    """Z-Curve representation."""
    features = []
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    for seq in sequences:
        seq = seq.upper()
        R = Y = M = K = W = S = 0
        x, y, z = [], [], []
        for nucle in seq:
            if nucle in ("A", "G"):
                R += 1
            else:
                Y += 1
            x.append(R - Y)

            if nucle in ("A", "C"):
                M += 1
            else:
                K += 1
            y.append(M - K)

            if nucle in ("A", "T", "U"):
                W += 1
            else:
                S += 1
            z.append(W - S)

        concat = x + y + z
        padding = (max_length - len(seq)) * 3
        mapping = np.pad(concat, (0, padding), 'constant')
        features.append(mapping)

    return pd.DataFrame(features, index=[f"seq{i+1}" for i in range(len(sequences))])


def integer(sequences, max_length=None):
    """Integer representation."""
    features = []
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    mapping_dict = {"T": 0, "U": 0, "C": 1, "A": 2, "G": 3}
    for seq in sequences:
        seq = seq.upper()
        mapping = [mapping_dict.get(n, 3) for n in seq]
        padding = max_length - len(mapping)
        mapping = np.pad(mapping, (0, padding), 'constant')
        features.append(mapping)

    return pd.DataFrame(features, index=[f"seq{i+1}" for i in range(len(sequences))])


def real(sequences, max_length=None):
    """Real number representation."""
    features = []
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    mapping_dict = {"T": 1.5, "U": 1.5, "C": 0.5, "A": -1.5, "G": -0.5}
    for seq in sequences:
        seq = seq.upper()
        mapping = [mapping_dict.get(n, -0.5) for n in seq]
        padding = max_length - len(mapping)
        mapping = np.pad(mapping, (0, padding), 'constant')
        features.append(mapping)

    return pd.DataFrame(features, index=[f"seq{i+1}" for i in range(len(sequences))])


def EIIP(sequences, max_length=None):
    """EIIP representation."""
    features = []
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    mapping_dict = {"T": 0.1335, "U": 0.1335, "C": 0.1340, "A": 0.1260, "G": 0.0806}
    for seq in sequences:
        seq = seq.upper()
        mapping = [mapping_dict.get(n, 0.0806) for n in seq]
        padding = max_length - len(mapping)
        mapping = np.pad(mapping, (0, padding), 'constant')
        features.append(mapping)

    return pd.DataFrame(features, index=[f"seq{i+1}" for i in range(len(sequences))])


def atomic_number(sequences, max_length=None):
    """Atomic number representation."""
    features = []
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    mapping_dict = {"T": 66, "U": 66, "C": 58, "A": 70, "G": 78}
    for seq in sequences:
        seq = seq.upper()
        mapping = [mapping_dict.get(n, 78) for n in seq]
        padding = max_length - len(mapping)
        mapping = np.pad(mapping, (0, padding), 'constant')
        features.append(mapping)

    return pd.DataFrame(features, index=[f"seq{i+1}" for i in range(len(sequences))])
