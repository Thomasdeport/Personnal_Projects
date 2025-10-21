import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx

from load import load_data

class CFMGraphDataset(Dataset):
    def __init__(
        self,
        test_size=0.1,
        seed=42,
        dummy=False,
    ):
        """Dataset générant directement des graphes en mémoire sans stockage .pt"""
        self.test_size = test_size
        self.seed = seed
        self.dummy = dummy

        # Chargement unique des données en RAM
        self.X, self.y, self.X_test = load_data(dummy=self.dummy, graph=True)

        # Gestion des splits
        train_idx, val_idx = train_test_split(
            np.arange(len(self.X)), test_size=self.test_size, random_state=self.seed
        )
        
        self.splits = {
            "train": train_idx,
            "val": val_idx,
            "test": np.arange(len(self.X_test))
        }
        
        self.data_cache = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if idx in self.data_cache:
            return self.data_cache[idx]
        
        data_idx = self.index[idx]
        graph_data = self._process_single_graph(data_idx)
        
        self.data_cache[idx] = graph_data
        return graph_data

    def _process_single_graph(self, idx):
        """Génère un graphe à partir des données brutes sans stockage intermédiaire."""
        X_i = self.X[idx] if idx < len(self.X) else self.X_test[idx - len(self.X)]
        label = torch.tensor(self.y[idx], dtype=torch.long) if self.y is not None else torch.tensor(-1, dtype=torch.long)

        graph = nx.Graph()
        node_features = np.zeros((len(X_i), 11), dtype=np.float32)
        venues_previous_keys = {}
        order_ids_keys = {}

        for k, row in enumerate(X_i):
            order_id = row[1]
            venue = row[0]
            graph.add_node(k)
            node_features[k, :] = np.array(
                [venue, k, row[5], row[8], row[9], row[10], row[11], row[12], row[13],row[14],row[15]],
                dtype=np.float32,
            )
            edge_attr = np.array([row[2] == 0, row[2] == 1, row[2] == 2, row[3], row[4]], dtype=np.float32)
            
            if venue in venues_previous_keys:
                graph.add_edge(k, venues_previous_keys[venue], edge_attr=edge_attr)
            venues_previous_keys[venue] = k
            
            if order_id in order_ids_keys:
                graph.add_edge(k, order_ids_keys[order_id][-1], edge_attr=edge_attr)
                order_ids_keys[order_id].append(k)
            else:
                order_ids_keys[order_id] = [k]
        
        data = from_networkx(graph)
        data.x = torch.from_numpy(node_features)
        data.y = label
        return data

    def set_split(self, split):
        """Définit l'indexation des données en fonction du split demandé."""
        self.index = self.splits[split]