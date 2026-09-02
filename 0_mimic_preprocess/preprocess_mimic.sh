docker run --rm \
  -v "${PWD}/mimic:/app/input" \
  -v "${PWD}/0_mimic_preprocess/preprocessed_file:/app/output" \
  -v "${PWD}/0_mimic_preprocess/features:/app/features" \
  -v "${PWD}/0_mimic_preprocess/extdata:/app/extdata" \
  -v "${PWD}/config.ini:/app/config/config.ini" \
  -v "${PWD}/panel_name_to_feature_codes.py:/app/panel_name_to_feature_codes.py:ro" \
  mimic-preprocessor
