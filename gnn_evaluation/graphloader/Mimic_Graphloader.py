
import torch
from torch_geometric.loader import NeighborLoader
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from constants.feature_names import FEATURES
from gnn_evaluation.utils.SeedInitialization import seed_worker

class GraphLoaderMimic:
    def __init__(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_graph = None
        self.test_graph = None
        self.train_edge_index = None
        self.train_edge_weight = None
        self.test_edge_index = None
        self.test_edge_weight = None
        self.pos_encodings = None
        self.test_pos_encodings = None
        self.df = None
        self.test_df = None
        self.labels = None
        self.test_labels = None

    def load_nodes(self):
        self.df = pd.read_csv("./data/graph_data/mimic_sorted_processed.csv", header=0)
        self.test_df = pd.read_csv("./data/graph_data/mimic_test_sorted_processed.csv", header=0)
        self.labels = self.df["Label"].astype('category').cat.codes.astype(float)
        self.test_labels = self.test_df["Label"].astype('category').cat.codes.astype(float)

    def load_edges(self):
        self.edge_index = torch.load("./data/graph_data/mimic_edge_index.pt") 
        self.edge_weight = torch.load("./data/graph_data/mimic_edge_weight.pt")
        self.test_edge_index = torch.load("./data/graph_data/mimic_test_edge_index.pt") 
        self.test_edge_weight = torch.load("./data/graph_data/mimic_test_edge_weight.pt")

    def load_pos_encodings(self):
        pos_encodings = torch.load("./data/graph_data/mimic_pos_encodings.pt")
        test_pos_encodings = torch.load("./data/graph_data/mimic_test_pos_encodings.pt")
        self.pos_encodings = pos_encodings
        self.test_pos_encodings = test_pos_encodings

    def scale_features(self):
        scaler = StandardScaler()
        self.df[FEATURES] = scaler.fit_transform(self.df[FEATURES])
        self.test_df[FEATURES] = scaler.transform(self.test_df[FEATURES])   

    def construct_graph(self):
        self.train_graph = Data(x = torch.tensor(self.df[FEATURES].values, dtype=torch.float),
                            edge_index = self.edge_index.long(),
                            edge_attr = self.edge_weight.float(),
                            y = torch.tensor(self.labels, dtype=torch.float).unsqueeze(1))
        self.test_graph = Data(x = torch.tensor(self.test_df[FEATURES].values, dtype=torch.float),
                            edge_index = self.test_edge_index.long(),
                            edge_attr = self.test_edge_weight.float(),
                            y = torch.tensor(self.test_labels, dtype=torch.float).unsqueeze(1))

    def construct_train_val_test_masks(self):
        train_ids, val_ids = train_test_split(self.df["Id"].unique(), test_size=0.2, random_state=42)
        train_mask = self.df["Id"].isin(train_ids)
        val_mask = self.df["Id"].isin(val_ids)
        self.train_graph.train_mask = torch.tensor(train_mask.values, dtype=torch.bool)
        self.train_graph.test_mask = torch.tensor(val_mask.values, dtype=torch.bool)
        self.test_graph.test_mask = torch.tensor(self.test_graph.num_nodes * [True], dtype=torch.bool) 
    
    def add_pos_encodings(self):
        self.train_graph.x = self.train_graph.x + self.pos_encodings
        self.test_graph.x = self.test_graph.x + self.test_pos_encodings

    def initialize_graphs(self):
        self.load_edges()
        self.load_pos_encodings()
        self.load_nodes()
        self.scale_features()

        self.construct_graph()
        self.construct_train_val_test_masks()
        self.add_pos_encodings()
    
    def initialize(self, batch_size):
        self.initialize_graphs()
        g = torch.Generator()
        g.manual_seed(42)
        self.train_loader = NeighborLoader(self.train_graph, num_neighbors=[-1] * 2, batch_size=batch_size, input_nodes=self.train_graph.train_mask, generator=g, worker_init_fn=seed_worker)
        self.val_loader = NeighborLoader(self.train_graph, num_neighbors=[-1] * 2, batch_size=batch_size, input_nodes=self.train_graph.test_mask, generator=g, worker_init_fn=seed_worker)
        self.test_loader = NeighborLoader(self.test_graph, num_neighbors=[-1] * 2, batch_size=batch_size, input_nodes=self.test_graph.test_mask, generator=g, worker_init_fn=seed_worker)