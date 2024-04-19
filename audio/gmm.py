import librosa
import numpy as np

from glob import glob
from ikrlib import train_gmm, logpdf_gmm
from librosa.feature import mfcc
from pathlib import Path
from numpy.random import randint
from audiomentations import ApplyImpulseResponse, AddBackgroundNoise, PolarityInversion

SAMPLE_RATE = 16000
GMM_COMPONENTS = 16
ITERATIONS = 25


def augment_data(name: str, waveform: np.ndarray) -> list[(str, np.ndarray)]:
    """
    Augment data.
    """
    augmented = []

    # Keep the original recording
    augmented.append((name, waveform))

    # Augment adding Gaussian noise
    augmented.append(
        (f"{name}_gaussian_noise", waveform + np.random.normal(0, 0.1, waveform.shape))
    )

    # Augment using the time stretching (change of speed)
    rnd_speed = np.random.uniform(0.7, 1.3)
    augmented.append(
        (f"{name}_speed", librosa.effects.time_stretch(y=waveform, rate=rnd_speed))
    )

    # Augment adding Room Impulse Response
    rir_augment = ApplyImpulseResponse(
        ir_path="RIRS_NOISES/real_rirs_isotropic_noises", p=1.0
    )
    augmented.append((f"{name}_rir", rir_augment(waveform, sample_rate=SAMPLE_RATE)))

    # Augment adding background noise
    background_noise = AddBackgroundNoise(
        sounds_path="RIRS_NOISES/pointsource_noises",
        min_snr_in_db=3.0,
        max_snr_in_db=30.0,
        noise_transform=PolarityInversion(p=0.5),
        p=1.0,
    )
    augmented.append(
        (
            f"{name}_background_noise",
            background_noise(waveform, sample_rate=SAMPLE_RATE),
        )
    )

    return augmented


def load_recordings(
    dir_name: str, n_mfcc=13, initial_pause=1.7, augmentation=False
) -> list[tuple[str, np.ndarray]]:
    """
    Load all *.wav files sampled at `SAMPLE_RATE` from directory `dir_name`, convert them into MFCC features and
    return them as array of tuples [(segment_name, mfcc)].

    First `INITIAL_PAUSE` seconds of each recording are removed.
    """
    waveforms = []
    for f in glob(dir_name + "/*.wav"):
        segment_name = Path(f).stem
        waveform, sr = librosa.load(f, sr=None)
        assert sr == SAMPLE_RATE

        waveforms.append((segment_name, waveform[int(initial_pause * sr) :]))

    augmented = []
    if augmentation:
        for segment_name, waveform in waveforms:
            augmented.extend(augment_data(segment_name, waveform))
        waveforms = augmented

    mfccs = []
    for s, w in waveforms:
        mfccs.append((s, mfcc(y=w, sr=SAMPLE_RATE, n_mfcc=n_mfcc)))

    return mfccs


def evaluate(
    recordings: list[(str, np.ndarray)],
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    nweights: np.ndarray,
    nmeans: np.ndarray,
    ncovs: np.ndarray,
    p_t=0.5,
    p_nt=0.5,
):
    for segment_name, features in recordings:
        posterior_t = sum(logpdf_gmm(features.T, weights, means, covs)) + np.log(p_t)
        posterior_nt = sum(logpdf_gmm(features.T, nweights, nmeans, ncovs)) + np.log(
            p_nt
        )

        score = posterior_t - posterior_nt
        decision = 1 if (score > 0) else 0
        print(f"{segment_name} {score} {decision}")


if __name__ == "__main__":
    # Training
    targets = load_recordings("data/target_train", augmentation=True)
    non_targets = load_recordings("data/non_target_train", augmentation=True)

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
        print(
            f"Total log likelihood for target class: {tll}; for non-target class: {tll2}"
        )

    # Evaluation
    recordings = load_recordings("data/target_dev")
    evaluate(recordings, weights, means, covs, nweights, nmeans, ncovs)
    recordings = load_recordings("data/non_target_dev")
    evaluate(recordings, weights, means, covs, nweights, nmeans, ncovs)
