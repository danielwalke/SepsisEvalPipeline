docker run --rm `
  -v "${PWD}/1_preprocess/data/preprocessed_data:/app/input" `
  -v "${PWD}/3_graph_construction/data:/app/output" `
  3_graph_construction