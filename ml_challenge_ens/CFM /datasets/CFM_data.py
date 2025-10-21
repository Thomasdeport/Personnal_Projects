
from load import load_data
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class CFMDataset(Dataset):
    def __init__(
        self,
        split="train",
        test_size=0.1,
        seed=42,
        dummy=False
    ):
        """Dataset traitant les données brutes sans créer de graphes"""
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.dummy = dummy

        # Chargement des données
        self.X, self.y, self.X_test = load_data(dummy=self.dummy, graph=False)

        # Gestion des splits
        if self.split == "test":
            self.X = self.X_test
            self.y = None
            self.index = list(range(len(self.X)))
        elif self.split == "train":
            self.index, _ = train_test_split(
                list(range(len(self.X))), test_size=self.test_size, random_state=self.seed
            )
        elif self.split == "val":
            _, self.index = train_test_split(
                list(range(len(self.X))), test_size=self.test_size, random_state=self.seed
            )
        else:
            raise ValueError(f"Unknown split {self.split}")

        self.data_cache = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if idx in self.data_cache:
            return self.data_cache[idx]
        
        data_idx = self.index[idx]
        data = self._process_single_data(data_idx)
        
        # Mémorisation des données en cache pour une récupération rapide
        self.data_cache[idx] = data
        return data

    def _process_single_data(self, idx):
        """Retourne les données brutes, sans création de graphes"""
        X_i = self.X[idx] if idx < len(self.X) else self.X_test[idx - len(self.X)]
        label = torch.tensor(self.y[idx], dtype=torch.long) if self.y is not None else torch.tensor(-1, dtype=torch.long)
        
        # Ici on ne génère pas de graphe, mais on retourne simplement les données
        # Les données sont retournées sous forme de tenseur
        data = torch.tensor(X_i, dtype=torch.float32)
        
        return data, label
