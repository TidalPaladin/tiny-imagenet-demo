#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

tensorboard --logdir=/artifacts/tblogs &
exec "$@"
