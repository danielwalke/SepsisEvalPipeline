docker run --rm `
  -v "${PWD}/1_preprocess/data/preprocessed_data:/app/input" `
  -v "${PWD}/2_baseline:/app/output" `
  -v "${PWD}/config.ini:/app/config/config.ini" `
  2_baseline