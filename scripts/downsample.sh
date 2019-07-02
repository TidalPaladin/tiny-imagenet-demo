#!/bin/bash
# Downsamples an input image to $OUT_RES
#-unsharp 1.5x1+0.7+0.02 \

downsample() {
	src=$(readlink -f $1 || exit 1)
	[ -f ${src} ] || exit 1

	dest=$2
	touch ${dest} || exit 1

	OUT_RES="64x64"
	DENSITY=96

	convert \
		${src} \
		-filter Mitchell \
		-sampling-factor "1x1" \
		-quality 98 \
		-resize ${OUT_RES}! \
		-density ${DENSITY} \
		${dest}

	exit $?
}
export -f downsample

if [ $# -eq 0 ]; then
	downsample $1 $2
fi
