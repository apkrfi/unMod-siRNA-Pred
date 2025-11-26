import numpy as np
import pandas as pd
from igraph import Graph

def _patterns(seq, k):
    """Yield all k-mers (sliding window)."""
    seqlen = len(seq)
    for i in range(seqlen):
        j = seqlen if i + k > seqlen else i + k
        yield seq[i:j]
        if j == seqlen:
            break


def _extract_features_from_graph(g: Graph):
    
    feats = {
        "avg_betweenness": np.mean(g.betweenness(directed=False, weights=None)),
        "avg_degree": np.mean(g.degree()),
        "assortativity_degree": g.assortativity_degree(directed=False),
        "max_degree": max(g.degree()),
        "min_degree": min(g.degree()),
        "std_degree": np.std(g.degree()),
        "avg_path_length": g.average_path_length(directed=False, unconn=False),
        "transitivity_local": g.transitivity_avglocal_undirected(),
        "transitivity_global": g.transitivity_undirected(),
        "num_edges": g.ecount(),
        "motifs_size3": g.motifs_randesu_no(size=3),
        "motifs_size4": g.motifs_randesu_no(size=4),
    }
    return feats


def complex_network_features(sequences, k=3, threshold=3):
    """
    Returns Complex Network features(by _extract_features_from_graph()) for sequences.

    Parameters
    ----------
    sequences : list[str]
       A list of RNA sequences.
    k: int
       k for k-mers
    threshold: int
       Threshold for eliminating edges of size < threshold to extract features again.

    Returns
    -------
    pandas.DataFrame
        DF with CTD features.

    """    
    all_features = []

    for seq in sequences:
        seq = seq.upper()
        # Build graph from k-mers
        cn = Graph()
        kmers = list(_patterns(seq, k))
        seen = set()

        for i in range(len(kmers) - 1):
            v1, v2 = kmers[i], kmers[i+1]
            if v1 not in seen:
                cn.add_vertices(v1); seen.add(v1)
            if v2 not in seen:
                cn.add_vertices(v2); seen.add(v2)
            cn.add_edges([(v1, v2)])

        # Extract features under thresholding
        seq_feats = {}
        for t in range(1, threshold):
            mat = np.array(cn.get_adjacency().data)
            mat = np.where(mat < t, 0, mat)
            g_thr = Graph.Adjacency(mat.tolist(), mode="undirected")

            if g_thr.ecount() < 1:
                # fill zeros if graph is too sparse
                for feat in ["avg_betweenness","avg_degree","assortativity_degree",
                             "max_degree","min_degree","std_degree",
                             "avg_path_length","transitivity_local","transitivity_global",
                             "num_edges","motifs_size3","motifs_size4"]:
                    seq_feats[f"{feat}_{t}"] = 0
            else:
                feats = _extract_features_from_graph(g_thr)
                for key, val in feats.items():
                    seq_feats[f"{key}_{t}"] = val

        all_features.append(seq_feats)

    return pd.DataFrame(all_features, index=[f"seq{i+1}" for i in range(len(sequences))])
