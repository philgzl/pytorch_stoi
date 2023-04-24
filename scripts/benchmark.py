import argparse
import itertools
import timeit
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_stoi import NegSTOILoss


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        x_len = x.shape[-1]
        return self.conv(x.view(-1, 1, x_len)).view(x.shape)


def plot():
    times_master = np.load(f'npy/times/master-{suffix}.npy')
    times_philgzl = np.load(f'npy/times/philgzl-{suffix}.npy')
    os.makedirs('plots/times', exist_ok=True)
    for j, wav_length in enumerate(args.wav_length):
        plt.figure()
        for times, label in [
            (times_master[:, j], 'master'),
            (times_philgzl[:, j], 'philgzl'),
        ]:
            plt.plot(times, 'o-', label=label)
        plt.title(
            f'repeats={args.repeats}, '
            f'wav_length={wav_length} s, '
            f'use_vad={use_vad} s, '
            f'extended={extended} s, '
            f'fs={args.fs}, '
        )
        plt.xlabel('Batch size')
        plt.ylabel('Time (s)')
        plt.xticks(range(len(args.batch_size)), args.batch_size)
        plt.legend()
        plt.savefig(f'plots/times/{suffix}-wav_length={wav_length}.png')

    plt.figure()
    for j, wav_length in enumerate(args.wav_length):
        plt.plot(
            times_philgzl[:, j] / times_master[:, j],
            'o-',
            label=f'wav_length={wav_length} s',
        )
        plt.title(
            f'repeats={args.repeats}, '
            f'use_vad={use_vad} s, '
            f'extended={extended} s, '
            f'fs={args.fs}, '
        )
    plt.xticks(range(len(args.batch_size)), args.batch_size)
    plt.xlabel('Batch size')
    plt.ylabel('Time fold increase (x)')
    plt.legend()
    plt.savefig(f'plots/times/diff_{suffix}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['master', 'philgzl'])
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--batch-size', type=int, nargs='+',
                        default=[1, 2, 4, 8])
    parser.add_argument('--wav-length', type=int, nargs='+',
                        default=[1, 2, 4, 8],)
    parser.add_argument('--fs', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    os.makedirs('npy/times', exist_ok=True)
    for use_vad, extended in itertools.product([True, False], [True, False]):
        suffix = (
            f'use_vad={use_vad}_'
            f'extended={extended}'
        )
        print(suffix)
        if args.benchmark is not None:
            outfile = f'npy/times/{args.benchmark}-{suffix}.npy'
            times = np.empty((len(args.batch_size), len(args.wav_length)))
            for i, j in itertools.product(
                range(len(args.batch_size)),
                range(len(args.wav_length)),
            ):
                batch_size = args.batch_size[i]
                wav_length = args.wav_length[j]
                print(f'batch_size={batch_size}, wav_length={wav_length}')
                criterion = NegSTOILoss(sample_rate=args.fs, use_vad=use_vad,
                                        extended=extended)
                np.random.seed(42)
                nnet = TestNet()
                x = torch.randn(batch_size, wav_length*args.fs)
                x[:, round(x.shape[-1]*0.5):round(x.shape[-1]*0.6)] = 0
                if args.cuda:
                    x = x.cuda()
                    nnet = nnet.cuda()
                    criterion = criterion.cuda()
                y = nnet(x)

                def to_time():
                    loss = criterion(x, y).mean()
                    loss.backward(retain_graph=True)

                times[i, j] = timeit.timeit(
                    "to_time()",
                    number=args.repeats,
                    setup=(
                        "from __main__ import args, x, y, criterion, to_time;"
                        "from pystoi import stoi"
                    )
                )
            np.save(outfile, times)

        if args.plot:
            plot()

    if args.show:
        plt.show()
