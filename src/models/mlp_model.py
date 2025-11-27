import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPVulnerabilityDetector(nn.Module):
    
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.3):
        super(MLPVulnerabilityDetector, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


def create_mlp_model(input_dim=512, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.3):
    return MLPVulnerabilityDetector(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    )
