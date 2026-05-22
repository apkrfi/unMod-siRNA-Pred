import pandas as pd
import os
import numpy as np
from basic import *
from chaos import *
from numerical_mappings import *
from physico_chemical import *
from CN import *
from config import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate RNA sequence representations')
    parser.add_argument(
        '--representations',
        nargs='+',
        choices=[
            'global_descriptor', 'one_hot', 'kmer_counts', 'GC_count', 'zcurve',
            'integer', 'atomic_number', 'EIIP', 'complex_network_features',
            'classifical_chaos', 'frequency_chaos', 'real'
        ],
        default=[
            'global_descriptor', 'one_hot', 'kmer_counts', 'GC_count', 'zcurve',
            'integer', 'atomic_number', 'EIIP', 'complex_network_features',
            'classifical_chaos', 'frequency_chaos', 'real'
        ],
        help='List of representations to generate (default: all)'
    )
    parser.add_argument(
        '--datapath',
        type=str,
        required=True,
        help='Path to the input CSV file containing siRNA sequences'
    )
    parser.add_argument(
        '--representation_dir',
        type=str,
        required=True,
        help='Directory to save the output representation CSV files'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.representation_dir, exist_ok=True)

    data = list(pd.read_csv(args.datapath)['siRNA_seq'])

    representation_funcs = {
        'global_descriptor': global_descriptor,
        'one_hot': one_hot,
        'kmer_counts': kmer_counts,
        'GC_count': GC_count,
        'zcurve': zcurve,
        'integer': integer,
        'atomic_number': atomic_number,
        'EIIP': EIIP,
        'complex_network_features': complex_network_features,
        'classifical_chaos': classifical_chaos,
        'frequency_chaos': frequency_chaos,
        'real': real,
    }

    for r in args.representations:
        print(f"Generating representation: {r}")
        func = representation_funcs[r]
        df = func(data)
        out_path = os.path.join(args.representation_dir, r + '.csv')
        df.to_csv(out_path)
        print(f"  Saved to: {out_path}")


if __name__ == '__main__':
    main()