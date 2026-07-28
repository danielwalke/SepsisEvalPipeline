import pandas as pd
import torch
from torch_geometric.utils import degree

import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

df = pd.read_csv("/home/daniel.walke/git/SepsisEvalPipeline/3_graph_construction/data/CBC/mimic_train_edges.csv")

print(df.head())
edge_index = torch.stack([torch.from_numpy(df["source"].to_numpy()), torch.from_numpy(df["target"].to_numpy())])
print(edge_index)