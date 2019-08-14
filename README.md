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

1. [Tensorflow 2.0](https://www.tensorflow.org/beta/)
2. [Abseil (for logging / flags)](https://abseil.io/docs/python/)

These dependencies can be installed with

```
pip install tensorflow-gpu==2.0.0-beta1 absl-py
```

or in Google Colab with

```
!pip install tensorflow-gpu==2.0.0-beta1 absl-py
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

If using the Docker image, start the container with

```shell
docker-compose up tiny-imagenet
```

to launch the container as a daemon (no log output).

This will automatically launch Tensorboard, which should now be
accessible by visiting `http://localhost:6006`.


Next, start the training script as follows:

```shell
docker exec -it tiny-imagenet python /app/train.py
```

You may optionally supply command line flags as follows:

```shell
docker exec -it tiny-imagenet python /app/train.py --batch_size=64
```
