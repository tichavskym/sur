# SUR

People detection based on voice recordings and head shot images

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

To launch training of Gaussian Mixture Model (GMM) for speaker recognition, run the following command:

```sh
python audio/gmm.py
```
