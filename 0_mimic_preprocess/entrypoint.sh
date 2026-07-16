#!/bin/bash
set -e
python3 /app/extract.py "$@"
Rscript /app/process_files.R
