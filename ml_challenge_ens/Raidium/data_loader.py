import numpy as np
import torch
from torch.utils.data import Dataset

class SupervisedCTScanDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        mask = self.masks[idx].astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = torch.tensor(img).unsqueeze(0)  # Grayscale: (1, H, W)
            mask = torch.tensor(mask, dtype=torch.long)

        return img, mask

class UnsupervisedCTScanDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            img = torch.tensor(img).unsqueeze(0)
        return img, idx  # On peut garder l'index pour matcher plus tard les pseudo-labels

class PseudoLabeledDataset(Dataset):
    def __init__(self, images, pseudo_masks, transform=None):
        self.images = images
        self.pseudo_masks = pseudo_masks
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        mask = self.pseudo_masks[idx].astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = torch.tensor(img).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.long)

        return img, mask


def One_hot_encoding(annotated, label_nb, y_train, annotations): 
    y_array = [y_train.iloc[n].unique() for n in range (annotated)]
    annotations_subset = annotations[:annotated]
    y_oh = np.zeros((annotated, label_nb-1))
    for picture in range(annotated): 
        for i in range(label_nb-1): 
            if i in y_array[picture] and i in annotations_subset.iloc[picture]: 
                y_oh[picture, i] = 1
            elif i not in y_array[picture] and i in annotations_subset.iloc[picture]: 
                y_oh[picture, i] = 0
            else: 
               y_oh[picture, i] = np.mean(y_oh[picture]) 
    return y_oh           
import torch 

def criterion_cls_with_nan(criterion_cls, labels, cls_out): 
    mask = ~torch.isnan(labels)  
    labels_clean = torch.nan_to_num(labels, nan=0.0)  
    loss_raw = criterion_cls(cls_out, labels_clean)  
    loss_cls = (loss_raw * mask).sum() / mask.sum()  #  ignore  NaN in the loss
    return loss_cls