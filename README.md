# SUR

People detection based on voice recordings and headshot images

## Development setup

To install the required dependencies, run the following command:

```sh
make venv
```

Also, download the dataset [from URL](https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2023-2024/) and store
it into `data/` directory.

If you want to use audio augmentation, download [Room Impulse Response and Noise Database](https://www.openslr.org/28/)
and extract it into `RIRS_NOISES/` directory. 

## Usage

To train and evaluate Gaussian Mixture Model (GMM) for speaker recognition, run the following commands:

```sh
# Train your model
python audio/gmm.py train gmm_model.npz

# Evaluate your model
# WARNING: never load models from untrusted sources, loading of the model is not secure against erroneous 
# or maliciously constructed data (uses pickle.load under the hood)
python audio/gmm.py eval gmm_model.npz
```

## Audio training evaluation

Evaluation of the GMM models during training was performed using `audio/peekin.py` and `plotting.py` helper scripts.
See following figure for performance of the GMM models on test and validation datasets with respect to the number of 
components and iterations used.

![GMM performance](doc/gmm_errors.png)
