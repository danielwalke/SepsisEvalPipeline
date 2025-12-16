docker run --rm `
  -v "D:\Datasets\mimic-iv-2.2:/app/input" `
  -v "${PWD}/preprocessed_file:/app/output" `
  -v "${PWD}/features:/app/features" `
  mimic-preprocessor