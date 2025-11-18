import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.03_dataset_pyg import VulnDataset
from src.04_model import GCNVuln

def train_model(samples, epochs=20, batch_size=16, lr=1e-3):
    dataset = VulnDataset(samples)
    train_loader = DataLoader(dataset[:int(0.8*len(dataset))], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[int(0.8*len(dataset)):], batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNVuln(in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss={total_loss/len(train_loader):.4f}")

    return model, val_loader
