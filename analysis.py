import argparse
import numpy as np
import matplotlib.pyplot as plt

__version__ = '0.1'

plt.rcParams.update(
    {
        'font.size': 30,
        'text.usetex': True,
        'figure.figsize': (14,11),
        'font.family': 'Computer Modern Roman'
    }
)

def main() -> None:
    _vals_times = [0, 1_000, 10_000, 100_000, 1_000_000]
    sklpy_times = [0, 0.003521, 0.034318, 0.535300, 8.836155]
    scipy_times = [0, 0.027732, 0.166335, 1.474478, 14.32770]
    vnlpy_times = [0, 0.101650, 1.221879, 20.48854, 191.4288]
    vnlcc_times = [0, 0.008865, 0.032988, 0.254328, 2.671155]
    eigcc_times = [0, 0.008202, 0.034422, 0.294001, 3.425298]
    cuda1_times = [0, 0.009314, 0.041698, 0.090047, 0.652350]
    cuda2_times = [0, 0.006274, 0.029613, 0.053129, 0.215813]

    plt.plot(_vals_times, sklpy_times, color='b', lw=4, label=r'{\sc Scikit-Learn}')
    plt.plot(_vals_times, scipy_times, color='k', lw=4, label=r'{\sc SciPy}')
    plt.plot(_vals_times, vnlpy_times, color='y', lw=4, label=r'{\sc Vanilla Python}')
    plt.plot(_vals_times, vnlcc_times, color='c', lw=4, label=r'{\sc Vanilla C{\tt++} std}')
    plt.plot(_vals_times, eigcc_times, color='m', lw=4, label=r'{\sc Vanilla C{\tt++} eigen}')
    plt.plot(_vals_times, cuda1_times, color='g', lw=4, label=r'{\sc Cuda v1}')
    plt.plot(_vals_times, cuda2_times, color='r', lw=4, label=r'{\sc Cuda v2}')
    plt.xticks(
        [0,200000,400000,600000,800000,1000000],
        ['0','200K','400K','600K','800K','1M']
    )
    plt.xlabel(r'Dataset Size')
    plt.ylabel(r'Runtime in Seconds')
    plt.ylim([-1, 15])
    plt.grid(linestyle='dotted', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
