from torch import nn
from torch_geometric.nn import GATConv
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, out_dim, non_lin = F.relu, heads=4, dropout=0.0, norm = nn.LayerNorm):
        super(Model, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, edge_dim=edge_dim, concat=True, heads=heads, add_self_loops=False, dropout=dropout)
        self.norm1 = norm(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, edge_dim=edge_dim, concat=False, heads=heads, add_self_loops=False, dropout=dropout)
        self.non_lin = non_lin
        self.dropout = dropout

    def forward(self, x, edge_index, **kwargs):
        edge_weight = kwargs.get('edge_weight', None)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x)
        x = self.non_lin(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


        
    