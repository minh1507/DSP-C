import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset, Data
from src.02_ast_graph import get_ast_graph

class VulnDataset(InMemoryDataset):
    def __init__(self, samples):
        super().__init__()
        self.data_list = []
        for s in samples:
            G = get_ast_graph(s['code'], s['ext'])
            data = self.nx_to_pyg(G, s['label'])
            self.data_list.append(data)
        self.data, self.slices = self.collate(self.data_list)

    def nx_to_pyg(self, G, label:int):
        # map node types → index
        node_types = list(nx.get_node_attributes(G, 'type').values())
        type2idx = {t:i for i,t in enumerate(set(node_types))}
        x = torch.tensor([[type2idx[t]] for t in node_types], dtype=torch.float)
        data = from_networkx(G)
        data.x = x
        data.y = torch.tensor([label], dtype=torch.long)
        return data
