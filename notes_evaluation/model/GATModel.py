from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim = 0, non_lin = F.relu, heads=4, dropout=0.0, norm = nn.LayerNorm, num_layers=3):
        super(GATModel, self).__init__()
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=edge_dim))
        self.layer_norms.append(norm(hidden_dim * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, edge_dim=edge_dim))
            self.layer_norms.append(norm(hidden_dim * heads))
        self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout, edge_dim=edge_dim))
        self.dropout = dropout
        self.non_lin = non_lin

    def forward(self, x, edge_index, **kwargs):
        batch = kwargs.get('batch', None)
        for conv, layer_norm in zip(self.convs[:-1], self.layer_norms):
            x = conv(x, edge_index)
            x = layer_norm(x)
            x = self.non_lin(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        return x.squeeze(-1)