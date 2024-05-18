import argparse
import numpy as np
import sklearn.datasets

__version__ = '0.1'

np.random.seed(42)

def uint(s: str) -> int:
    try: v = int(s)
    except ValueError: raise argparse.ArgumentTypeError(f'expected integer, got {s!r} instead.')
    if v <= 0: raise argparse.ArgumentTypeError(f'expected positive integer, got {v} instead.')
    return v

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
        epilog='Author: Filemon Mateus (mateus@utah.edu).',
        description='Generates synthetic clustering datasets for benchmarking.'
    )
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help='Show this help message and exits.'
    )
    parser.add_argument(
        '-v', '--version', action='version',
        version=f'%(prog)s {__version__}', default=argparse.SUPPRESS,
        help='Prints version information and exits.'
    )
    parser.add_argument(
        '--num_samples', type=uint, default=100,
        help='Number of samples to generate. Defaults to %(default)s.'
    )
    parser.add_argument(
        '--num_clusters', type=uint, default=3,
        help='Number of clusters to generate. Defaults to %(default)s.'
    )
    parser.add_argument(
        '--output_file', type=str, default='data.txt',
        help='Name of the output file to write the data to. Defaults to \'%(default)s\'.'
    )
    args = parser.parse_args()
    return args

def save_data(data: np.ndarray, labels: np.ndarray, outfile: str) -> None:
    with open(outfile, 'w') as file:
        for point, label in zip(data, labels):
            file.write(','.join(map(str, point)) + f',{label}\n')

def main() -> None:
    args = parse_args()
    data, labels = sklearn.datasets.make_blobs(
        n_samples=args.num_samples,
        centers=args.num_clusters
    )
    save_data(data, labels, args.output_file)

if __name__ == '__main__':
    main()
