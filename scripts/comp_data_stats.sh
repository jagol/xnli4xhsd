#!/bin/bash

DATA_DIR="/srv/scratch0/jgoldz/xnli-for-multilingual-hate-speech-detection/data"
PYTHON_SCRIPT="src/compute_data_stats.py"

find "${DATA_DIR}" -type f -name "*test_*.jsonl" -o -name "MHC_*_test_*.jsonl" | while IFS= read -r test_file; do
    output_json="${test_file%.jsonl}_stats.json"
    echo "Processing ${test_file}..."
    python3 "${PYTHON_SCRIPT}" -i "${test_file}" -o "${output_json}"
    echo "Output saved to ${output_json}"
done
