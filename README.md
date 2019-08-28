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

TODO Put a model.summary() call or other graphic here

The model implementation is based on the Resnet family of residual
vision networks and parameterized, allowing for easy adjustment of
depth and width.

TODO Put some metrics about how well it does on old Tiny ImageNet or
custom Tiny ImageNet.

The following model attributes are parameterized:

1. Whether to use a default `7x7/2` tail, user supplied tail, or no tail.
2. Whether to use a default head, user supplied head, or no head.
3. The number of output classes (if using the default head).
4. The number of downsampling levels, and the number of bottleneck layers within each level.
5. The model width (by supplying a custom tail).
