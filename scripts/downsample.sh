#!/bin/sh
file_in=$(readlink -f "$1")
file_out=$(readlink -f "$2")
OUT_RES="64x64"
DENSITY=96

convert \
	${file_in} \
	-filter Lanczos \
	-sampling-factor "1x1" \
	-quality 90 \
	-resize ${OUT_RES}! \
	-density ${DENSITY} \
	-unsharp 1.5x1+0.7+0.02 \
	"${file_out}"
