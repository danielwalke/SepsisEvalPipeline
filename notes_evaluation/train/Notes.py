import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, to_undirected
from notes_evaluation.model.GATModel import GATModel
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from gnn_evaluation.model.ModelWrapper import ModelWrapperClassifications
from sklearn.metrics import roc_auc_score
import tqdm

class NotesEvaluation:
    def __init__(self):
        self.graph_data_path = "./notes_evaluation/graph_data/"
        self.graphs = []
        self.train_graphs = []
        self.val_graphs = []
        self.test_graphs = []
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_graphs(self):
        for file in tqdm.tqdm(os.listdir(self.graph_data_path)):
            if not file.endswith(".pt"): continue
            graph = torch.load(os.path.join(self.graph_data_path, file), weights_only=False)
            graph.edge_index = to_undirected(graph.edge_index)
            self.graphs.append(graph)

    def split_graphs(self):
        labels = [graph.y.item() for graph in self.graphs]
        train_val_graphs, self.test_graphs = train_test_split(self.graphs, test_size=0.2, stratify=labels, random_state=42)
        train_val_labels = [graph.y.item() for graph in train_val_graphs]
        self.train_graphs, self.val_graphs = train_test_split(train_val_graphs, test_size=0.25, stratify=train_val_labels, random_state=42)  # 0.25 x 0.8 = 0.2
        
    def create_dataloaders(self, batch_size=32):
        self.train_loader = DataLoader(self.train_graphs, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_graphs, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_graphs, batch_size=batch_size, shuffle=False)

    def train_model(self):
        best_params = {
            'hidden_dim': 128,
            'heads': 8,
            'dropout': 0.00,
            'lr': 0.0005
        }
        model = GATModel(in_dim=self.train_graphs[0].num_features, hidden_dim=int(best_params['hidden_dim']), edge_dim=1, out_dim=1, heads=int(best_params['heads']), dropout=best_params['dropout'], non_lin=F.relu)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
        train_labels = torch.tensor([graph.y.item() for graph in self.train_graphs], device=self.device)
        pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=self.device))
            
        model_wrapper = ModelWrapperClassifications(model=model, model_name="GAT_SBC", epochs=5000, patience=5, pred_proba_based_metric=True, minimize_metric=False, eval_metric=roc_auc_score)
        model_wrapper.train(self.train_loader, self.val_loader, optimizer, criterion, self.device)
        test_auc = model_wrapper.evaluate(self.test_loader, self.device)
        print(f"Test AUC: {test_auc:.4f}")

    

if __name__ == "__main__":
    ne = NotesEvaluation()
    ne.load_graphs()
    ne.split_graphs()
    ne.create_dataloaders(batch_size=1028)
    ne.train_model()