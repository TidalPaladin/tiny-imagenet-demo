#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

python /app/train.py $@
