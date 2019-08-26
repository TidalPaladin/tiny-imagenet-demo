# Dataset Scripts

This directory contains scripts to aid in generating a downsampled
subset of the ImageNet dataset.

## Prerequisites

* Docker Compose
* Docker
* ImageNet training set

## Usage

Edit the `.env` file in this directory to assign the source /
destination filepaths as follows.

```shell
SRC_DIR=/path/to/train
DEST_DIR=/path/to/output
```

Build the Docker image:

```shell
docker-compose build
```

Finally, run the Docker image to begin dataset generation:

```shell
docker-compose run mk_tiny_imagenet
```


## Components

* `downsample.sh file_in file_out` uses ImageMagick to downsample an
	input file to `64x64` resolution without cropping.
* `group.sh src_dir dest_dir` groups classes in `src_dir` based on the N'th
	leftmost class label digits (ie `n1234` where N=4), sampling
	uniformly from each class in a group. This script also applies
	downsampling via `downsample.sh`. **This is the main script.**
