docker run --rm `
  -v "${PWD}/0_mimic_preprocess:/app/input" `
  -v "${PWD}/1_preprocess/data/preprocessed_data:/app/output" `
  -v "${PWD}/config.ini:/app/config/config.ini" `
  1_datapreprocess