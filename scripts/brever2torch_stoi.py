import argparse
import os

import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm
import pandas as pd

from brever.data import BreverDataset


def main():
    out_path = 'data/wav'
    os.makedirs(out_path, exist_ok=True)
    dataset = BreverDataset(args.in_path)
    filenames = []
    for i in tqdm(range(len(dataset))):
        noisy, clean = dataset[i].mean(axis=-2)
        filename = f'{out_path}/{i:03d}_16kHz.wav'
        torchaudio.save(filename.replace('.wav', '_est.wav'), noisy.unsqueeze(0), 16000)
        torchaudio.save(filename, clean.unsqueeze(0), 16000)
        filenames.append(filename)

        noisy = resample(noisy, 16000, 8000)
        clean = resample(clean, 16000, 8000)
        filename = f'{out_path}/{i:03d}_8kHz.wav'
        torchaudio.save(filename.replace('.wav', '_est.wav'), noisy.unsqueeze(0), 8000)
        torchaudio.save(filename, clean.unsqueeze(0), 8000)
        filenames.append(filename)

    df = pd.DataFrame(filenames, columns=['filename'])
    df.to_csv('data/filenames.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path')
    args = parser.parse_args()
    main()
