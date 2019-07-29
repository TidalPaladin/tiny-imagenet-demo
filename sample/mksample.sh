#!/bin/sh
# Downsamples an input image to $OUT_RES

file_in=$(readlink -f "$1")
OUT_RES="64x64"
DENSITY=96

for filter in "Lanczos" "Mitchell" "Castrom"
do
	file_out="$PWD/${filter}.JPEG"
	convert \
		${file_in} \
		-filter Lanczos \
		-sampling-factor "1x1" \
		-quality 98 \
		-resize ${OUT_RES}! \
		-density ${DENSITY} \
		"${file_out}"
	echo $(ls -hs $file_out)

	file_out="$PWD/${filter}_unsharp.JPEG"
	convert \
		${file_in} \
		-filter Lanczos \
		-sampling-factor "1x1" \
		-quality 98 \
		-resize ${OUT_RES}! \
		-density ${DENSITY} \
		-unsharp 1.5x1+0.7+0.02 \
		"${file_out}"
	echo $(ls -hs $file_out)
done
