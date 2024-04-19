import librosa
import numpy as np
import os.path

from glob import glob
from ikrlib import train_gmm
from librosa.feature import mfcc
from numpy.random import randint

SAMPLE_RATE = 16000
GMM_COMPONENTS = 16
ITERATIONS = 10


def load_recordings(dir_name: str, n_mfcc=13, initial_pause=1.7) -> list[tuple[str, np.ndarray]]:
    """
    Load all *.wav files sampled at `SAMPLE_RATE` from directory `dir_name`, convert them into MFCC features and
    return them as array of tuples [(segment_name, mfcc)].

    First `INITIAL_PAUSE` seconds of each recording are removed.
    """
    mfccs = []
    for f in glob(dir_name + '/*.wav'):
        segment_name = os.path.basename(f)
        waveform, sr = librosa.load(f, sr=None)
        assert sr == SAMPLE_RATE
        waveform = waveform[int(initial_pause * sr):]
        mfccs.append((segment_name, mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)))

    return mfccs


if __name__ == '__main__':
    targets = load_recordings("data/target_train")
    non_targets = load_recordings("data/non_target_train")

    targets = [x for _, x in targets]
    non_targets = [x for _, x in non_targets]
    targets = np.concatenate(targets, axis=1).T
    non_targets = np.concatenate(non_targets, axis=1).T

    # Initiate means, covariances, and weights for GMM
    means = targets[randint(0, len(targets), GMM_COMPONENTS)]
    avg_cov = np.cov(targets.T, bias=True)
    covs = np.array([avg_cov] * GMM_COMPONENTS)
    weights = np.ones(GMM_COMPONENTS) / GMM_COMPONENTS

    nmeans = non_targets[randint(0, len(non_targets), GMM_COMPONENTS)]
    navg_cov = np.cov(non_targets.T, bias=True)
    ncovs = np.array([navg_cov] * GMM_COMPONENTS)
    nweights = np.ones(GMM_COMPONENTS) / GMM_COMPONENTS

    for i in range(ITERATIONS):
        weights, means, covs, tll = train_gmm(targets, weights, means, covs)
        nweights, nmeans, ncovs, tll2 = train_gmm(non_targets, nweights, nmeans, ncovs)
        print(f"Total log likelihood for target class: {tll}; for non-target class: {tll2}")
