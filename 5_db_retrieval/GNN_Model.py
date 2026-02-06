from torch import nn
from torch_geometric.nn import GATv2Conv

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=1, dropout=0.3, heads = 4, activation=nn.ReLU(), num_layers=3, skip_connections=False):
        super().__init__()
        assert isinstance(in_channels, int) and in_channels > 0, "Input channels must be a positive integer"
        assert isinstance(hidden_channels, int) and hidden_channels > 0, "Hidden channels must be a positive integer"
        assert isinstance(out_channels, int) and out_channels > 0, "Output channels must be a positive integer"
        assert isinstance(dropout, float) and 0 <= dropout < 1, "Dropout must be a float in the range [0, 1)"
        assert isinstance(heads, int) and heads > 0, "Heads must be a positive integer"
        assert isinstance(num_layers, int) and num_layers >= 2, "Number of layers must be an integer greater than or equal to 2"
        assert isinstance(skip_connections, bool), "Skip connections must be a boolean value"
        
        self.bn_input = nn.BatchNorm1d(in_channels)
        
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=1, concat=True, residual = False) # First layer does not have skip connections, as it changes the feature dimension. Subsequent layers can have skip connections if specified.
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)  
        self.conv_layers.append(self.conv1)
        self.batch_norm_layers.append(self.bn1)
        
        for i in range(2, num_layers):
            conv = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=1, concat=True, residual =skip_connections)
            self.conv_layers.append(conv)
            self.batch_norm_layers.append(nn.BatchNorm1d(hidden_channels * heads))
        
        
        self.conv_out = GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=0, edge_dim=1, concat=False, residual = False) # Output layer does not use dropout or skip connections
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_attr=None):
        x = self.bn_input(x)
        
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.batch_norm_layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.conv_out(x, edge_index, edge_attr)
        return x
