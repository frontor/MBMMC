#!/usr/bin/env bash
set -euo pipefail
python examples/make_toy_data.py
bash examples/run_rf.sh
bash examples/run_crossnn.sh
bash examples/run_mpcnet.sh
echo "All smoke-training examples completed."
