import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

def to_loader (X,y,batch_size,val_size): 
    X_tensor = torch.tensor(X, dtype=torch.float32)  
    y_tensor = torch.tensor(y, dtype=torch.float32)    

    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=val_size, stratify=y_tensor, random_state=42
    )
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader
# # REMOVE OUTLIERS per stock in the traning data
#     X_train = X_train.merge(y_train, on = 'obs_id')
#     group_col = 'eqt_code_cat'
#     cols_to_check = ['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux']
#     if filter:
#         X_train = remove_outliers_per_cat(df = X_train , group_col = group_col, cols = cols_to_check , threshold=7)

#     y_train = X_train[["obs_id", 'eqt_code_cat']].reset_index()
#     y_train = y_train.drop_duplicates(subset='obs_id', keep='first')

#     X_train.drop(columns=[ 'eqt_code_cat'], inplace=True)