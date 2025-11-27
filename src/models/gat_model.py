import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATVulnerabilityDetector(nn.Module):
    
    def __init__(self, num_features=100, hidden_dim=64, num_heads=8, num_layers=3, 
                 num_classes=2, dropout=0.3):
        super(GATVulnerabilityDetector, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, 
                                         heads=num_heads, dropout=dropout, concat=True))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads,
                                         heads=num_heads, dropout=dropout, concat=True))
        
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        x = self.input_proj(x)
        x = F.relu(x)
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x
    
    def forward_from_adj(self, node_features, adj_matrix):
        batch_size, num_nodes, num_features = node_features.shape
        
        outputs = []
        for i in range(batch_size):
            edge_index = torch.nonzero(adj_matrix[i] > 0, as_tuple=False)
            if edge_index.size(0) == 0:
                edge_index = torch.arange(num_nodes, device=node_features.device).repeat(2, 1)
            else:
                edge_index = edge_index.t()
            
            x = node_features[i]
            output = self.forward(x, edge_index, batch=None)
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)
    
    def predict(self, x, edge_index=None, adj_matrix=None, batch=None):
        self.eval()
        with torch.no_grad():
            if adj_matrix is not None:
                logits = self.forward_from_adj(x, adj_matrix)
            else:
                logits = self.forward(x, edge_index, batch)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


def create_gat_model(num_features=100, hidden_dim=64, num_heads=8, num_layers=3, 
                    num_classes=2, dropout=0.3):
    return GATVulnerabilityDetector(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )
