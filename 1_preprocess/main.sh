docker run --rm \
  -v "$(pwd)/0_mimic_preprocess:/app/input" \
  -v "$(pwd)/1_preprocess/data/preprocessed_data:/app/output" \
  -v "$(pwd)/config.ini:/app/config/config.ini" \
  1_datapreprocess

