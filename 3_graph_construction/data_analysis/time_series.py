import pandas as pd
import os
import torch
from torch_geometric.utils import degree

def get_number_of_samples(nodes_df):
    return nodes_df.shape[0]

def get_indegree_distribution(edges_df, num_nodes):
    target_indices = torch.from_numpy(edges_df["target"].to_numpy())
    deg = degree(target_indices, num_nodes=num_nodes)
    return deg

def get_median_indegree(edges_df, num_nodes):
    indegree_dist = get_indegree_distribution(edges_df, num_nodes)
    median_indegree = indegree_dist.median().item()
    return median_indegree

train_nodes_cbc = pd.read_csv("../data/CBC/mimic_train_nodes.csv")
train_edges_cbc = pd.read_csv("../data/CBC/mimic_train_edges.csv")
train_nodes_cbc_bmp_hil = pd.read_csv("../data/CBC_BMP_HIL/mimic_train_nodes.csv")
train_edges_cbc_bmp_hil = pd.read_csv("../data/CBC_BMP_HIL/mimic_train_edges.csv")

train_edges_cbc_edge_index = torch.stack([torch.from_numpy(train_edges_cbc["source"].to_numpy()), torch.from_numpy(train_edges_cbc["target"].to_numpy())])

print("CBC Graph Degree Distribution:")
print(get_indegree_distribution(train_edges_cbc, get_number_of_samples(train_nodes_cbc)))
print("CBC Graph Median Indegree:")
print(get_median_indegree(train_edges_cbc, get_number_of_samples(train_nodes_cbc)))

print("CBC_BMP_HIL Graph Degree Distribution:")
print(get_indegree_distribution(train_edges_cbc_bmp_hil, get_number_of_samples(train_nodes_cbc_bmp_hil)))
print("CBC_BMP_HIL Graph Median Indegree:")
print(get_median_indegree(train_edges_cbc_bmp_hil, get_number_of_samples(train_nodes_cbc_bmp_hil)))