#!/bin/sh
CONVERT="$PWD/downsample.sh"
THREADS=6

# Examples per class in output dataset
CLASS_SIZE=1000

# Read source dir from args or use current directory
SRC_DIR=$(readlink -f "$1" || echo "$PWD")
[ -d $SRC_DIR ] || (echo "Source dir doesn't exist"; exit 1)

# Read dest dir from args or use current directory + /dataset
DEST_DIR=$(readlink -f "$2" || echo "$PWD/dataset")
mkdir -p $DEST_DIR || (echo "Could not create dest dir"; exit 1)

# TODO remove overrides before release
SRC_DIR="/mnt/valak/documents/imagenet/ILSVRC/Data/CLS-LOC/train"
DEST_DIR="/home/tidal/Documents/tiny-imagenet-demo/dataset"

CLASSES=$(readlink -f "$PWD/class_list.txt")
[ -f "$CLASSES" ] || (echo "Could not find class_list.txt"; exit 1)
lines=$(grep -oE "n[0-9]*" $CLASSES)

# Group classes up to the 4th numeric digit in class labels
GROUP_DIGIT=4
prefixes=$(echo "$lines" | grep -oE "n[0-9]{$GROUP_DIGIT}" | uniq)
echo "Matched $(wc -l <<< $prefixes) prefixes"

for prefix in $prefixes
do

	# Class list matching current prefix
	classes=$(grep -oE "${prefix}[0-9]*" $CLASSES)
	num_classes=$(echo "$classes" | wc -l)
	echo "Processing $prefix: $num_classes classes"

	# How many examples to take from each class
	take=$(($CLASS_SIZE / $num_classes))

	for class in $classes
	do
		dir=$(readlink -f $SRC_DIR/$class)
		[ -d $dir ] || (echo "$dir does not exist" && continue)

		# Make directory for examples aggregated by prefix
		dest="$DEST_DIR/$prefix"
		mkdir -p $dest || (echo "Could not create $prefix output dir"; continue)

		find $dir/ | head -n $take | xargs -r -I {} -P $THREADS \
			sh -c "$CONVERT {} $dest/\$(basename {})" \
			&& echo "Processed $take examples from $class"
	done
	echo "Finished prefix $prefix"
done
