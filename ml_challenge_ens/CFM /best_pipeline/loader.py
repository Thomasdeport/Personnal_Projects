import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset


def get_loaders(loaded_data, batch_size=32, test_size=0.1, seed=42, shuffle = True):
    X, y, X_test = loaded_data

    # Gestion des splits
    X_train, X_val, y_train, y_val = train_test_split(
        X,y, test_size=test_size,stratify=y, random_state=seed
    )

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    
    return train_loader,val_loader,test_loader