#!/bin/bash
set -e

if [ ! -f /app/input/hosp/labevents.csv ] || [ ! -f /app/input/hosp/d_labitems.csv ]; then
    cat >&2 <<'MSG'
================================================================================
 MIMIC-IV data not found.

 This pipeline requires the MIMIC-IV "hosp" module, which needs a free
 PhysioNet credentialed-access account:
   1. Request access:  https://physionet.org/content/mimiciv/
   2. Documentation:    https://mimic.mit.edu/docs/iv/
   3. Download the "hosp" folder and place its CSV files under:
        ./mimic/hosp/labevents.csv, ./mimic/hosp/d_labitems.csv, ...
      (docker-compose.yml mounts ./mimic as /app/input, so this must
      exist on the host before running `docker compose up`)
================================================================================
MSG
    exit 1
fi

mkdir -p /app/extdata
cp /app/input/hosp/d_labitems.csv /app/extdata/d_labitems.csv

PANEL_CONFIG_PATH=/app/config/config.ini \
PANEL_FEATURES_DIR=/app/features \
PANEL_WRITE_ENV=0 \
python3 /app/panel_name_to_feature_codes.py
python3 /app/extract.py "$@"
Rscript /app/process_files.R
