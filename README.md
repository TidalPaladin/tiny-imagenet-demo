# Tiny ImageNet Demo

This library implements a vision classification network written in
Tensorflow 2.0 for Tiny ImageNet. The intention is to provide a
demonstration of Tensorflow 2.0's high level Keras APIs for
implementing models and training pipelines.

Additionally, scripts are included to generate a downsampled version
of ImageNet (Tiny ImageNet) that is suitable for training on commodity
hardware or under time constraints.

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
pip install tensorflow-gpu==2.0.0-rc0 absl-py
```

or in Google Colab with

```shell
!pip install tensorflow-gpu==2.0.0-rc0 absl-py
```

The following dependencies are optional:
1. [Tensorboard (for visualization)](https://www.tensorflow.org/tensorboard/)

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
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
tail (Tail)                  (None, 64, 64, 32)        4704
_________________________________________________________________
bottleneck (Bottleneck)      (None, 64, 64, 32)        744
_________________________________________________________________
bottleneck_1 (Bottleneck)    (None, 64, 64, 32)        744
_________________________________________________________________
bottleneck_2 (Bottleneck)    (None, 64, 64, 32)        744
_________________________________________________________________
downsample (Downsample)      (None, 32, 32, 64)        20432
_________________________________________________________________
bottleneck_3 (Bottleneck)    (None, 32, 32, 64)        2512
_________________________________________________________________
bottleneck_4 (Bottleneck)    (None, 32, 32, 64)        2512
_________________________________________________________________
bottleneck_5 (Bottleneck)    (None, 32, 32, 64)        2512
_________________________________________________________________
bottleneck_6 (Bottleneck)    (None, 32, 32, 64)        2512
_________________________________________________________________
bottleneck_7 (Bottleneck)    (None, 32, 32, 64)        2512
_________________________________________________________________
bottleneck_8 (Bottleneck)    (None, 32, 32, 64)        2512
_________________________________________________________________
downsample_1 (Downsample)    (None, 16, 16, 128)       80800
_________________________________________________________________
bottleneck_9 (Bottleneck)    (None, 16, 16, 128)       9120
_________________________________________________________________
bottleneck_10 (Bottleneck)   (None, 16, 16, 128)       9120
_________________________________________________________________
bottleneck_11 (Bottleneck)   (None, 16, 16, 128)       9120
_________________________________________________________________
bottleneck_12 (Bottleneck)   (None, 16, 16, 128)       9120
_________________________________________________________________
downsample_2 (Downsample)    (None, 8, 8, 256)         321344
_________________________________________________________________
bn (BatchNormalization)      (None, 8, 8, 256)         1024
_________________________________________________________________
relu (ReLU)                  (None, 8, 8, 256)         0
_________________________________________________________________
head (TinyImageNetHead)      (None, 61)                15677
=================================================================
Total params: 497,765
Trainable params: 493,653
Non-trainable params: 4,112
_________________________________________________________________
```

TODO Put some metrics about how well it does on old Tiny ImageNet or
custom Tiny ImageNet.

The following model attributes are parameterized:

1. Whether to use a default `7x7/2` tail, user supplied tail, or no tail.
2. Whether to use a default head, user supplied head, or no head.
3. The number of output classes (if using the default head).
4. The number of downsampling levels, and the number of bottleneck layers within each level.
5. The model width (by supplying a custom tail).
