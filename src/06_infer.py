import torch
from torch_geometric.loader import DataLoader
from src.03_dataset_pyg import VulnDataset

def inference(model, samples):
    dataset = VulnDataset(samples)
    loader = DataLoader(dataset, batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
    return preds
