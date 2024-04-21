# SUR

People detection based on voice recordings and headshot images.

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

To train and evaluate the ResNet18 model for the person recognition, run the following commands:

```sh
# Train the model
python src/images_resnet/train_resnet.py --dataset /path/to/the/dataset/dir/

# Evaluate the model
python src/images_resnet/eval_resnet.py --model /path/to/models/model_checkpoint.pt --dataset /path/to/the/dataset/dir/

# Plot statistics
python3 src/images_resnet/plotting.py
```

## Audio training evaluation

Evaluation of the GMM models during training was performed using `audio/peekin.py` and `plotting.py` helper scripts.
See following figure for performance of the GMM models on training and validation datasets with respect to the number of 
components and iterations used.

![GMM performance](doc/gmm_errors.png)

The model we selected for further use is located at `models/gmm_audio_24_27.npz` (24 components, 27 iterations).

## Image training evaluation
See the following figure for the performance of the ResNet18 model (train and validation loss and validation accuracy).

![ResNet18 performance](doc/resnet_stats.png)

The model we selected for further use is located at `models/resnet_image_15_20.pt` (saved at 15 epochs of 20 epochs totally) chosen using the Early Stopping approach.
