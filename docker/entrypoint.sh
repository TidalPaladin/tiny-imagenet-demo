#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

TB_DIR=/artifacts/tblogs
echo "** Starting Tensorboard. Logdir=${TB_DIR} **"
tensorboard --logdir=/artifacts/tblogs &

if [[ "$*" = "/run.sh" ]]; then
	echo "** Starting default training pipeline **"
	exec "$@"
elif [[ ! -f "$1" ]]; then
	echo "** Starting default training pipeline with user flags **"
	exec "/run.sh" "$@"
else
	echo "** Starting custom training pipeline **"
	exec "$@"
fi
