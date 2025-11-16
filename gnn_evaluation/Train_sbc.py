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
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import NeighborLoader

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, edge_dim=1, concat=True, heads=heads, add_self_loops=False)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, edge_dim=1, concat=False, heads=heads, add_self_loops=False)
        # self.conv3 = GATConv(hidden_dim * heads, output_dim, edge_dim=1, heads=heads, concat=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.normalize(x, p=2., dim=-1)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        # x = F.elu(x)
        # x = self.conv3(x, edge_index, edge_weight)
        return x
    
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)[data.train_mask]
        loss = criterion(out, data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        labels = []
        pred_probas = []
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)[data.val_mask]
            pred_probas.extend(torch.sigmoid(out).cpu().numpy())
            labels.extend(data.y[data.val_mask].cpu().numpy())
        auc = roc_auc_score(labels, pred_probas)
    return auc

if __name__ == "__main__":
    edge_index = torch.load("./data/graph_data/sbc_edge_index.pt") 
    edge_weight = torch.load("./data/graph_data/sbc_edge_weight.pt")
    sbc_df = pd.read_csv("./data/graph_data/sbc_sorted_processed.csv", header=0)
    labels = sbc_df["Label"].astype('category').cat.codes.astype(float)
    ##CHeck sort?
    
    test_edge_index = torch.load("./data/graph_data/sbc_validation_edge_index.pt") 
    test_edge_weight = torch.load("./data/graph_data/sbc_validation_edge_weight.pt")
    sbc_validation_df = pd.read_csv("./data/graph_data/sbc_validation_sorted_processed.csv", header=0)
    validation_labels = sbc_validation_df["Label"].astype('category').cat.codes.astype(float)

    scaler = StandardScaler()
    sbc_df[FEATURES] = scaler.fit_transform(sbc_df[FEATURES])
    sbc_validation_df[FEATURES] = scaler.transform(sbc_validation_df[FEATURES])

    train_graph = Data(x = torch.tensor(sbc_df[FEATURES].values, dtype=torch.float),
                          edge_index = edge_index.long(),
                          edge_attr = edge_weight.float(),
                          y = torch.tensor(labels, dtype=torch.float).unsqueeze(1))
    test_graph = Data(x = torch.tensor(sbc_validation_df[FEATURES].values, dtype=torch.float),
                         edge_index = test_edge_index.long(),
                         edge_attr = test_edge_weight.float(),
                         y = torch.tensor(validation_labels, dtype=torch.float).unsqueeze(1))
    train_ids, val_ids = train_test_split(sbc_df["Id"].unique(), test_size=0.2, random_state=42)
    train_mask = sbc_df["Id"].isin(train_ids)
    val_mask = sbc_df["Id"].isin(val_ids)
    train_graph.train_mask = torch.tensor(train_mask.values, dtype=torch.bool)
    train_graph.val_mask = torch.tensor(val_mask.values, dtype=torch.bool)
    test_graph.val_mask = torch.tensor([True]*test_graph.num_nodes, dtype=torch.bool)

    train_pos_encodings = torch.load("./data/graph_data/sbc_pos_encodings.pt")
    test_pos_encodings = torch.load("./data/graph_data/sbc_validation_pos_encodings.pt")
    train_graph.x = train_graph.x + train_pos_encodings
    test_graph.x = test_graph.x + test_pos_encodings

    BATCH_SIZE = 512
    LR= 1e-3
    train_loader = NeighborLoader(train_graph, num_neighbors=[-1] * 2, batch_size=BATCH_SIZE, input_nodes=train_graph.train_mask)
    val_loader = NeighborLoader(train_graph, num_neighbors=[-1] * 2, batch_size=BATCH_SIZE, input_nodes=train_graph.val_mask)
    test_loader = NeighborLoader(test_graph, num_neighbors=[-1] * 2, batch_size=BATCH_SIZE, input_nodes=test_graph.val_mask)

    model = GNN(input_dim=len(FEATURES), hidden_dim=128, output_dim=1, heads=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,betas=(0.9, 0.999), eps=1e-08)
    train_labels = train_graph.y
    pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
    print(f"Positive class weight: {pos_weight}")
    loss_fun = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    best_model_state = None
    best_val_auc = float('-inf')
    PATIENCE = 5
    patience_counter = 0
    for epoch in range(1, 5_000 +1):
        loss = train(model, train_loader, optimizer, loss_fun, device)
        val_auc = evaluate(model, val_loader, device)
        test_auc = evaluate(model, test_loader, device)
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
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    auc = evaluate(model, test_loader, device)
    print(f"Test AUC: {auc:.4f}") # 