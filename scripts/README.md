# Dataset Scripts

This directory contains scripts to aid in generating a downsampled
subset of the ImageNet dataset.

* `downsample.sh file_in file_out` uses ImageMagick to downsample an
	input file to `64x64` resolution.
* `mkdataset.sh src_dir dest_dir` reads `drop.txt` and `group.txt` to
	select a subset of training examples which are then downsampled via
	`downsample.sh` to produce a final output dataset.

The `drop.txt` file contains a list of class labels (ie `n0123456`)
that will be excluded from the final dataset, one label per line.

The `group.txt` file contains class labels that should be grouped
together into a single class in the final dataset. Each line
represents a grouping, and all class labels on a single line will be
grouped together. When grouping `N` classes, the training examples
that make up the output class will be drawn uniformly over each of the
`N` original classes.
