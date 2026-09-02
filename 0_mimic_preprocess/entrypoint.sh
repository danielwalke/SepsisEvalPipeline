#!/bin/bash
set -e
PANEL_CONFIG_PATH=/app/config/config.ini \
PANEL_FEATURES_DIR=/app/features \
PANEL_WRITE_ENV=0 \
python3 /app/panel_name_to_feature_codes.py
python3 /app/extract.py "$@"
Rscript /app/process_files.R
