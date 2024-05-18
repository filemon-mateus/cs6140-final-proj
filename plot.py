import argparse
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from matplotlib.colors import ListedColormap

__version__ = '0.1'

plt.rcParams.update(
    {
        'font.size': 36,
        'text.usetex': True,
        'figure.figsize': (9,7),
        'font.family': 'Computer Modern Roman'
    }
)

cmap = ListedColormap([
    '#8dd3c7', '#bebada', '#fb8072', '#80b0d3', '#fdb562', '#b3de69'
])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
        epilog='Author: Filemon Mateus (mateus@utah.edu).',
        description='Plots synthetic clustering datasets.'
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
        help='Path to the input data file containing the data to be plotted.'
    )
    parser.add_argument(
        '--means_file', type=str,
        help='Path to the means file containing centroids to be plotted.'
    )
    args = parser.parse_args()
    return args

def load_data(data_file: str) -> np.ndarray:
    data = np.genfromtxt(data_file, delimiter=',')
    return data

def plot_data(data: np.ndarray, means: Optional[np.ndarray] = None) -> None:
    plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=cmap, alpha=0.8, s=10)
    if means is not None: plt.scatter(means[:,0], means[:,1], c='r', marker='x', s=600)
    plt.tight_layout()
    plt.show()

def main() -> None:
    args = parse_args()
    data = load_data(args.data_file)

    if args.means_file:
        means = load_data(args.means_file)
        plot_data(data, means)
        return

    plot_data(data)

if __name__ == '__main__':
    main()
