docker run --rm `
  -v "D:\Datasets\mimic-iv-2.2:/app/input" `
  -v "${PWD}/0_mimic_preprocess/preprocessed_file:/app/output" `
  -v "${PWD}/0_mimic_preprocess/features:/app/features" `
  mimic-preprocessor