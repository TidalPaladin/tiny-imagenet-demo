#!/bin/bash
CONVERT="${PWD}/downsample.sh"
THREADS=10
GROUP_DIGIT=4

export SHELL=$(type -p bash)
source ${CONVERT}

# Examples per class in output dataset
CLASS_SIZE=1000

# Read source dir from args or use current directory
SRC_DIR=$(readlink -f "$1" || echo "${PWD}")
[ -d ${SRC_DIR} ] || (echo "Source dir doesn't exist"; exit 1)

# Read dest dir from args or use current directory + /dataset
DEST_DIR=$(readlink -f "$2" || echo "${PWD}/dataset")
mkdir -p ${DEST_DIR} || (echo "Could not create dest dir"; exit 1)

CLASS_FILE=$(readlink -f "${PWD}/class_list.txt")
[ -f "${CLASS_FILE}" ] || (echo "Could not find class_list.txt"; exit 1)

# Group classes up to a numeric digit in class labels
prefixes=$(grep -oE "n[0-9]{$GROUP_DIGIT}" ${CLASS_FILE} | uniq)

for prefix in ${prefixes}
do

	# Class list matching current prefix
	classes=$(grep -oE "${prefix}[0-9]*" ${CLASS_FILE})
	num_classes=$(echo "$classes" | wc -l)
	echo "Processing $prefix: $num_classes classes"

	# How many examples to take from each class
	take=$((${CLASS_SIZE} / ${num_classes}))

	for class in ${classes}
	do
		# Look for the class directory in original dataset
		src=$(readlink -f ${SRC_DIR}/${class})
		[ -d ${src} ] || (echo "${src} does not exist"; continue)

		# Make dest directory for examples aggregated by prefix
		dest="${DEST_DIR}/${prefix}"
		mkdir -p ${dest} || (echo "Could not create ${prefix} output dir"; continue)

		ls ${src} | grep 'JPEG' | head -n ${take} | parallel \
			-r \
			-j $THREADS \
			downsample "$src/{}" "$dest/{}" || exit 2

		echo "Processed $take examples from $class"

	done
	echo "Finished prefix $prefix"
done
