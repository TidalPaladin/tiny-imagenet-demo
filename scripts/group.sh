#!/bin/bash

# Import downsample function
export SHELL=$(type -p bash)
source downsample.sh
DOWNSAMPLE_CMD="downsample"

THREADS=30
GROUP_DIGIT=3
CLASS_SIZE=1000

# Read source dir from args or use current directory
SRC_DIR=$(readlink -f "$1" || echo "${PWD}")
[ -d ${SRC_DIR} ] || (echo "Source dir doesn't exist"; exit 1)

# Read dest dir from args or use current directory + /dataset
DEST_DIR=$(readlink -f "$2" || echo "${PWD}/dataset")
mkdir -p ${DEST_DIR} || (echo "Could not create dest dir"; exit 1)

# Logging
LOG_FILE="${DEST_DIR}/output.log"
echo "Begin log file:" > ${LOG_FILE}
print_log() {
	printf '%s %s\n' "$(date -Iseconds)" "$1" | tee -a ${LOG_FILE}
}

# Group classes up to a numeric digit in class labels
prefixes=$(ls ${SRC_DIR} | grep -oE "n[0-9]{$GROUP_DIGIT}" | uniq)

# Print operation description
print_log "Source: ${SRC_DIR}"
print_log "Destination: ${DEST_DIR}"
print_log "Group by digit: ${GROUP_DIGIT}"
print_log "Class size: ${CLASS_SIZE}"
print_log "Output classes: $(echo ${prefixes} | wc -w)"

print_log "Starting dataset generation..."
for prefix in ${prefixes}
do

	# Class list matching current prefix
	classes=$(ls -1 ${SRC_DIR} | grep -E "${prefix}[0-9]*")
	num_classes=$(echo "$classes" | wc -l)

	# How many examples to take from each class
	take=$((${CLASS_SIZE} / ${num_classes}))

	# Make dest directory for examples aggregated by prefix
	dest=${DEST_DIR}/${prefix}
	mkdir -p ${dest} || (print_log "Could not create ${prefix} output dir"; exit 2)

	for class in ${classes}
	do
		class_path="${SRC_DIR}/${class}"
		line_fmt="%s\t${dest}/${class}_%d.JPEG"
		awk_cmd="{printf(\"${line_fmt}\\n\",\$1,NR-1)}"
		ls ${class_path}/*.JPEG | head -n ${take} | awk "${awk_cmd}"
	done
done | tee -a $LOG_FILE | parallel --bar --colsep '\t' -r -j $THREADS downsample '{1} {2}'
print_log "Finished!"
