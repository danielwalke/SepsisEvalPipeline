#!/bin/bash
set -e

echo "--- STEP 1: PREPROCESS ---"
docker build -f "${PWD}/1_preprocess/Dockerfile" -t 1_datapreprocess "${PWD}/1_preprocess"
docker run --rm \
  -v "${PWD}/0_mimic_preprocess:/app/input" \
  -v "${PWD}/1_preprocess/data/preprocessed_data:/app/output" \
  -v "${PWD}/config.ini:/app/config/config.ini" \
  1_datapreprocess

echo "--- STEP 2: BASELINE ---"
docker build -f "${PWD}/2_baseline/Dockerfile" -t 2_baseline "${PWD}/2_baseline" 
docker run --rm \
  -v "${PWD}/1_preprocess/data/preprocessed_data:/app/input" \
  -v "${PWD}/2_baseline:/app/output" \
  -v "${PWD}/config.ini:/app/config/config.ini" \
  2_baseline

echo "--- STEP 3: GRAPH CONSTRUCTION ---"
docker build -f "${PWD}/3_graph_construction/Dockerfile" -t 3_graph_construction "${PWD}/3_graph_construction"
docker run --rm \
  -v "${PWD}/1_preprocess/data/preprocessed_data:/app/input" \
  -v "${PWD}/3_graph_construction/data:/app/output" \
  3_graph_construction

echo "--- STEP 4: DATABASE UPLOAD ---"
docker compose -f ./4_db_upload/docker-compose.yml up --build --abort-on-container-exit

echo "--- STEPS 5 & 6: DB STARTUP & TRAINING ---"
docker compose -f ./6_graphaware/docker-compose.yml up --build --wait

python 5_gnn_training/main.py >> gnn_training.log

cd 6_graphaware
python true_mini_batch.py >> graphaware_training.log
cd ..

docker compose -f ./6_graphaware/docker-compose.yml down

echo "--- PIPELINE COMPLETE ---"