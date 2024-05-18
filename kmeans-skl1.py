import time
import argparse
import numpy as np
import sklearn.cluster

__version__ = '0.1'

def uint(s: str) -> int:
    try: v = int(s)
    except ValueError: raise argparse.ArgumentTypeError(f'expected integer, got {s!r} instead.')
    if v <= 0: raise argparse.ArgumentTypeError(f'expected positive integer, got {v} instead.')
    return v

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
        epilog='Author: Filemon Mateus (mateus@utah.edu).',
        description='Implements kmeans clustering with sklearn.'
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
        '--data_file', type=str, required=True,
        help='Path to the input data file containing the data to be clustered.'
    )
    parser.add_argument(
        '--num_clusters', type=uint, required=True,
        help='Number of clusters to generate.'
    )
    parser.add_argument(
        '--max_iter', type=uint, default=300,
        help='Maximum number of iterations. Defaults to %(default)s.'
    )
    parser.add_argument(
        '--num_trials', type=uint, default=5,
        help='Number of trials to perform clustering subroutine. Defaults to %(default)s.'
    )
    args = parser.parse_args()
    return args

def load_data(data_file: str) -> np.ndarray:
    data = np.genfromtxt(data_file, delimiter=',')
    return data

def dump_means(means: np.ndarray) -> None:
    for cluster in means:
        print(f'{cluster[0]},{cluster[1]}')

def main() -> None:
    args = parse_args()
    data = load_data(args.data_file)
    model = sklearn.cluster.KMeans(
        args.num_clusters,
        max_iter=args.max_iter,
        init='random',
        algorithm='full',
        tol=0,
        n_init=1
    )

    tic = time.perf_counter()
    for _ in range(args.num_trials):
        means = model.fit(data).cluster_centers_
    toc = time.perf_counter()

    dump_means(means)
    print(f'# runtime: {(toc-tic) / args.num_trials:.6f}')

if __name__ == '__main__':
    main()
