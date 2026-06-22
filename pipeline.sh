#!/bin/bash
set -e

python -m 0_mimic_preprocess.panel_name_to_feature_codes

docker build ./0_mimic_preprocess/. -t mimic-preprocessor

PANEL_NAME=$(awk -F '=[ ]*' '/^ *panel_name/ {print $2}' config.ini | tr -d ' "')
R_PREPROCESS_DIR="${PWD}/0_mimic_preprocess/preprocessed_file/${PANEL_NAME}"
PRE_PROCESS_DIR="${PWD}/1_preprocess/data/preprocessed_data/${PANEL_NAME}"
GRAPH_CONSTRUCTION_DIR="${PWD}/3_graph_construction/data/${PANEL_NAME}"
METRICS_DIR="${PWD}/3_graph_construction/metrics"

echo "PANEL_NAME: $PANEL_NAME, R_PREPROCESS_DIR: $R_PREPROCESS_DIR, PRE_PROCESS_DIR: $PRE_PROCESS_DIR, GRAPH_CONSTRUCTION_DIR: $GRAPH_CONSTRUCTION_DIR"

mkdir -p "$R_PREPROCESS_DIR"
mkdir -p "$PRE_PROCESS_DIR"
mkdir -p "$GRAPH_CONSTRUCTION_DIR"
mkdir -p "$METRICS_DIR"

docker run --rm \
  -v "${PWD}/mimic:/app/input" \
  -v "${R_PREPROCESS_DIR}:/app/output" \
  -v "${PWD}/0_mimic_preprocess/features:/app/features" \
  mimic-preprocessor

ls -l "$R_PREPROCESS_DIR"

echo "--- STEP 1: PREPROCESS ---"
docker build -f "${PWD}/1_preprocess/Dockerfile" -t 1_datapreprocess "${PWD}/1_preprocess"

docker run --rm \
  -v "${R_PREPROCESS_DIR}:/app/input" \
  -v "${PWD}/0_mimic_preprocess/features:/app/features" \
  -v "${PWD}/0_mimic_preprocess/extdata:/app/extdata" \
  -v "${PRE_PROCESS_DIR}:/app/output" \
  -v "${PWD}/config.ini:/app/config/config.ini" \
  1_datapreprocess

echo "--- STEP 2: BASELINE ---"
ls -l "$PRE_PROCESS_DIR"
docker build -f "${PWD}/2_baseline/Dockerfile" -t 2_baseline "${PWD}/2_baseline"
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "${PRE_PROCESS_DIR}:/app/input" \
  -v "${PWD}/2_baseline:/app/output" \
  -v "${PWD}/config.ini:/app/config/config.ini" \
  -v "${PWD}/2_baseline/models:/app/models" \
  2_baseline

echo "--- STEP 3: GRAPH CONSTRUCTION ---"
docker build -f "${PWD}/3_graph_construction/Dockerfile" -t 3_graph_construction "${PWD}/3_graph_construction"
docker run --rm \
  -v "${PRE_PROCESS_DIR}:/app/input" \
  -v "${GRAPH_CONSTRUCTION_DIR}:/app/output" \
  -v "${METRICS_DIR}:/app/metrics" \
  -v "${PWD}/config.ini:/app/config/config.ini" \
  3_graph_construction

echo "--- STEP 4: DATABASE UPLOAD ---"
GRAPH_DIR="${GRAPH_CONSTRUCTION_DIR}" docker compose -f ./4_db_upload/docker-compose.yml up --build --abort-on-container-exit

echo "--- STEPS 5 & 6: DB STARTUP & TRAINING ---"
docker compose -f ./5_gnn_training/docker-compose.yml up --build --wait
docker compose -f ./6_graphaware/docker-compose.yml up --build --wait


# python ./5_gnn_training/main.py >> gnn_training.log

# python ./6_graphaware/main.py >> graphaware_training.log
# python 6_graphaware/Interpretability.py 

# docker compose -f ./6_graphaware/docker-compose.yml down
echo "--- PIPELINE COMPLETE ---"