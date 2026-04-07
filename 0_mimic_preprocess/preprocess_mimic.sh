docker run --rm \
  -v "${PWD}/mimic:/app/input" \
  -v "${PWD}/0_mimic_preprocess/preprocessed_file:/app/output" \
  -v "${PWD}/0_mimic_preprocess/features:/app/features" \
  mimic-preprocessor
