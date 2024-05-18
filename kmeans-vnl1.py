import time
import argparse
import numpy as np

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
        description='Implements kmeans clustering in vanilla Python.'
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

def kmeans(data: np.ndarray, num_clusters: int, max_iter: int) -> np.ndarray:
    num_samples = data.shape[0]
    num_features = data.shape[1]
    initial_indices = np.random.choice(num_samples, num_clusters)
    means = data[initial_indices].T

    data_repeat = np.stack([data] * num_clusters, axis=-1)
    all_rows = np.arange(num_samples)
    all_zero = np.zeros([1,1,2])

    for _ in range(max_iter):
        distances = np.sum(np.square(data_repeat - means), axis=1)
        label = np.argmin(distances, axis=-1)
        sparse = np.zeros([num_samples, num_clusters, num_features])
        sparse[all_rows, label] = data
        counts = (sparse != all_zero).sum(axis=0)
        means = sparse.sum(axis=0).T / counts.clip(min=1).T

    return means.T

def load_data(data_file: str) -> np.ndarray:
    data = np.genfromtxt(data_file, delimiter=',')
    return data

def dump_means(means: np.ndarray) -> None:
    for cluster in means:
        print(f'{cluster[0]},{cluster[1]}')

def main() -> None:
    args = parse_args()
    data = load_data(args.data_file)
    data = data[:,:-1]

    tic = time.perf_counter()
    for _ in range(args.num_trials):
        means = kmeans(data, args.num_clusters, args.max_iter)
    toc = time.perf_counter()

    dump_means(means)
    print(f'# runtime: {(toc-tic) / args.num_trials:.6f}')

if __name__ == '__main__':
    main()
