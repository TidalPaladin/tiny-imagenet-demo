# Tiny ImageNet Demo

This library implements a vision classification network written in
Tensorflow 2.0 for Tiny ImageNet. The intention is to provide a
demonstration of Tensorflow 2.0's high level Keras APIs for
implementing models and training pipelines.

Additionally, scripts are included to generate a downsampled version
of ImageNet (Tiny ImageNet) that is suitable for training on commodity
hardware or under time constraints.

Model / training code can be found in the [tin](./tin) directory. TODO
maybe rename this path?

Some theoretical aspects of the network are discussed [here](./theory.ipynb)

## Prerequisites

A Dockerfile with necessary dependencies is provided. You may also
follow the instructions below for a manual install or to install
on [Google Colab](https://colab.research.google.com/).

The following dependencies are required:

1. [Tensorflow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs)
2. [Abseil (for logging / flags)](https://abseil.io/docs/python/)
3. Pillow (for `keras.ImageDataGenerator`)
4. Sci-py (for `keras.ImageDataGenerator`)

These dependencies can be installed with

```shell
pip install tensorflow-gpu==2.0.0-rc0 absl-py Pillow==6.1.0 scipy==1.3.1
```

or in Google Colab with

```shell
!pip install tensorflow-gpu==2.0.0-rc0 absl-py Pillow==6.1.0 scipy==1.3.1
```

The following dependencies are optional:
1. [Tensorboard (for visualization)](https://www.tensorflow.org/tensorboard/)

TODO May need extra install steps for nvidia-docker to enable
Dockerized GPU training. Explore this

## Installation

Installation is automated using docker-compose. Build the local
Docker image with

```
docker-compose build tiny-imagenet
```

This will trigger a pull of all needed dependencies.

## Usage

If using the Docker image, first build the image with

```shell
docker-compose build tiny-imagenet
```

Next, run the training pipeline with

```shell
docker-compose run tiny-imagenet
```

This will automatically launch Tensorboard in the container, which should now be
accessible by visiting `http://localhost:6006`.

You may optionally supply command line flags as follows:


```shell
docker-compose run tiny-imagenet --dry --levels=3,5,7
```

To see all available command line flags, run

```shell
docker-compose run tiny-imagenet --helpfull
```

## Model Overview

The model implementation is based on the Resnet family of residual
vision networks and parameterized, allowing for easy adjustment of
depth and width. The default model as is follows:

```
Model: "tiny_image_net"
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #
================================================================================
tail (Tail)                         (None, 64, 64, 32)              729
________________________________________________________________________________
bottleneck (Bottleneck)             (None, 64, 64, 32)              776
________________________________________________________________________________
bottleneck_1 (Bottleneck)           (None, 64, 64, 32)              776
________________________________________________________________________________
bottleneck_2 (Bottleneck)           (None, 64, 64, 32)              776
________________________________________________________________________________
downsample (Downsample)             (None, 32, 32, 64)              20336
________________________________________________________________________________
bottleneck_3 (Bottleneck)           (None, 32, 32, 64)              2576
________________________________________________________________________________
bottleneck_4 (Bottleneck)           (None, 32, 32, 64)              2576
________________________________________________________________________________
bottleneck_5 (Bottleneck)           (None, 32, 32, 64)              2576
________________________________________________________________________________
bottleneck_6 (Bottleneck)           (None, 32, 32, 64)              2576
________________________________________________________________________________
bottleneck_7 (Bottleneck)           (None, 32, 32, 64)              2576
________________________________________________________________________________
bottleneck_8 (Bottleneck)           (None, 32, 32, 64)              2576
________________________________________________________________________________
downsample_1 (Downsample)           (None, 16, 16, 128)             80608
________________________________________________________________________________
bottleneck_9 (Bottleneck)           (None, 16, 16, 128)             9248
________________________________________________________________________________
bottleneck_10 (Bottleneck)          (None, 16, 16, 128)             9248
________________________________________________________________________________
bottleneck_11 (Bottleneck)          (None, 16, 16, 128)             9248
________________________________________________________________________________
bottleneck_12 (Bottleneck)          (None, 16, 16, 128)             9248
________________________________________________________________________________
downsample_2 (Downsample)           (None, 8, 8, 256)               320960
________________________________________________________________________________
bn (BatchNormalization)             (None, 8, 8, 256)               1024
________________________________________________________________________________
relu (ReLU)                         (None, 8, 8, 256)               0
________________________________________________________________________________
head (TinyImageNetHead)             (None, 61)                      15677
================================================================================
Total params: 494,110
Trainable params: 490,446
Non-trainable params: 3,664
________________________________________________________________________________
```

Currently, the [strongest model](./models/20190919-040503) reaches 52%
validation accuracy and 76% top-5 validation accuracy using a modified
version of Inception-Resnet with no pretrained weights. The same model
converged at 94% training accuracy and 99.8% top-5 training accuracy.

Weights and a description of the architecture used for this model are
available at the same location. The high level architecture of the
inception-style model is as follows:

```
Model: "tiny_inception_net"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
inception_tail (InceptionTai (None, 64, 64, 64)        3387
_________________________________________________________________
incept_A_1 (InceptionResnetA (None, 64, 64, 64)        3664
_________________________________________________________________
incept_A_2 (InceptionResnetA (None, 64, 64, 64)        3664
_________________________________________________________________
reduct_A_1 (InceptionReducti (None, 32, 32, 192)       9600
_________________________________________________________________
incept_B_1 (InceptionResnetB (None, 32, 32, 192)       19824
_________________________________________________________________
incept_B_2 (InceptionResnetB (None, 32, 32, 192)       19824
_________________________________________________________________
incept_B_3 (InceptionResnetB (None, 32, 32, 192)       19824
_________________________________________________________________
incept_B_4 (InceptionResnetB (None, 32, 32, 192)       19824
_________________________________________________________________
reduct_B_2 (InceptionReducti (None, 16, 16, 768)       118272
_________________________________________________________________
incept_C_1 (InceptionResnetC (None, 16, 16, 768)       300480
_________________________________________________________________
incept_C_2 (InceptionResnetC (None, 16, 16, 768)       300480
_________________________________________________________________
bn (BatchNormalization)      (None, 16, 16, 768)       3072
_________________________________________________________________
relu (ReLU)                  (None, 16, 16, 768)       0
_________________________________________________________________
head (TinyImageNetHead)      (None, 200)               153800
=================================================================
Total params: 975,715
Trainable params: 966,851
Non-trainable params: 8,864
_________________________________________________________________
```

The following model attributes are parameterized:

1. Whether to use a default tail, user supplied tail, or no tail.
2. Whether to use a default head, user supplied head, or no head.
3. The number of output classes (if using the default head).
4. The number of downsampling levels, and the number of bottleneck layers within each level.
5. The model width.
6. Training hyperparameters (learning rate, lr decay, regularization)
7. The use of dropout before the FC layer in the head

More complex parameterizations are available for the inception style
variant and are documented in the relevant code.

## Dataset

TODO Add notes about dataset once final bugs are worked out. Currently
61 classes with 1000 examples each.
