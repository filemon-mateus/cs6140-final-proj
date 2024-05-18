import argparse
import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score

__version__ = '0.1'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
        epilog='Author: Filemon Mateus (mateus@utah.edu).',
        description='Quantifies the quality of centroids with permutation independent scores.'
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
        help='Path to the input data file containing the ground truth data.'
    )
    parser.add_argument(
        '--means_file', type=str, required=True,
        help='Path to the means file containing centroids to be scored.'
    )
    args = parser.parse_args()
    return args

def load_data(data_file: str) -> np.ndarray:
    data = np.genfromtxt(data_file, delimiter=',')
    return data

def query_labels(data: np.ndarray, means: np.ndarray) -> np.ndarray:
    labels = np.zeros(data.shape[0])
    for point_index, point in enumerate(data[:,:-1]):
        point_dists = np.linalg.norm(point - means, axis=1)
        labels[point_index] = np.argmin(point_dists)
    return labels

def main() -> None:
    args = parse_args()
    data = load_data(args.data_file)
    means = load_data(args.means_file)
    labels = query_labels(data, means)

    print(f'# accuracy: {adjusted_mutual_info_score(data[:,-1], labels):.6f}')
    print(f'# silhouette: {silhouette_score(data[:,:-1], labels, sample_size=1_000):.6f}')

if __name__ == '__main__':
    main()
