#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

exec "python" "/app/train.py" "$@"
