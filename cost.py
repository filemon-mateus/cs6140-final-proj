import argparse
import numpy as np

__version__ = '0.1'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
        epilog='Author: Filemon Mateus (mateus@utah.edu).',
        description='Quantifies the quality of centroids with permutation independent costs.'
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
        help='Path to the means file containing centroids whose cost are to be computed.'
    )
    parser.add_argument(
        '--cost_type', choices=['kmeans', 'kcenter', 'kmedians'], type=str, default='kmeans',
        help='Cost to be used for evaluating centroids. Defaults to "%(default)s" cost.'
    )
    args = parser.parse_args()
    return args

def load_data(data_file: str) -> np.ndarray:
    data = np.genfromtxt(data_file, delimiter=',')
    return data

def center_dists(data: np.ndarray, means: np.ndarray) -> np.ndarray:
    dists = np.zeros(data.shape[0])
    for point_index, point in enumerate(data[:,:-1]):
        point_dists = np.linalg.norm(point - means, axis=1)
        dists[point_index] = np.min(point_dists)
    return dists

def kmeans_cost(data: np.ndarray, means: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(center_dists(data, means))))

def kcenter_cost(data: np.ndarray, means: np.ndarray) -> float:
    return np.max(center_dists(data, means))

def kmedians_cost(data: np.ndarray, means: np.ndarray) -> float:
    return np.mean(center_dists(data, means))

def main() -> None:
    args = parse_args()
    data = load_data(args.data_file)
    means = load_data(args.means_file)

    if args.cost_type == 'kmeans':
        print(f'{kmeans_cost(data, means):.6f}')
    elif args.cost_type == 'kcenter':
        print(f'{kcenter_cost(data, means):.6f}')
    else:
        print(f'{kmedians_cost(data, means):.6f}')

if __name__ == '__main__':
    main()
