import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
import pandas as pd
from constants.feature_names import FEATURES
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GNN, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, edge_dim=1, concat=True, heads=heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, output_dim, edge_dim=1, concat=False, heads=heads)
        # self.conv3 = GATv2Conv(hidden_dim * heads, output_dim, edge_dim=1, heads=heads, concat=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_weight)
        # x = F.elu(x)
        # x = self.conv3(x, edge_index, edge_weight)
        return x
    
def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)[data.train_mask]
    loss = criterion(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)[data.test_mask]
        pred_probas = torch.sigmoid(out).cpu().numpy()
        labels = data.y[data.test_mask].cpu().numpy()
        auc = roc_auc_score(labels, pred_probas)
    return auc

if __name__ == "__main__":
    edge_index = torch.load("./data/graph_data/mimic_edge_index.pt") 
    edge_weight = torch.load("./data/graph_data/mimic_edge_weight.pt")
    mimic_df = pd.read_csv("./data/graph_data/mimic_sorted_processed.csv", header=0)
    labels = mimic_df["Label"].astype('category').cat.codes.astype(float)
    ## TODO stratify?
    train_ids, test_ids = train_test_split(mimic_df["Id"].unique(), test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
    train_mask = mimic_df["Id"].isin(train_ids)
    val_mask = mimic_df["Id"].isin(val_ids)
    test_mask = mimic_df["Id"].isin(test_ids)
    train_mask = torch.tensor(train_mask.values, dtype=torch.bool)
    val_mask = torch.tensor(val_mask.values, dtype=torch.bool)
    test_mask = torch.tensor(test_mask.values, dtype=torch.bool)

    model = GNN(input_dim=len(FEATURES), hidden_dim=32, output_dim=1, heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)
    ## TODO weighted loss
    train_labels = labels[train_mask.numpy()]
    pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
    loss_fun = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = torch.tensor(mimic_df[FEATURES].values, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
    edge_index = edge_index.long()
    edge_weight = edge_weight.float()
    graph = Data(x=features, edge_index=edge_index, edge_attr=edge_weight, y=labels, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)
    graph = graph.to(device)
    model = model.to(device)
    best_model_state = None
    best_val_auc = float('-inf')
    PATIENCE = 50
    patience_counter = 0
    for epoch in range(1, 1_001):
        loss = train(model, graph, optimizer, loss_fun, device)
        val_auc = evaluate(model, graph, device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            model.load_state_dict(best_model_state)
            print("Early stopping triggered")
            break
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")
    auc = evaluate(model, graph, device)
    print(f"Test AUC: {auc:.4f}") # 0.8389