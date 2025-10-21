import os
import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
class CTScanDatasetManager:
    def __init__(self, data_dir, image_shape=(256, 256)):
        self.data_dir = Path(data_dir)
        self.image_shape = image_shape

        # Chargement des masques et des annotations
        self.y_train = pd.read_csv(self.data_dir / "y_train.csv", index_col=0).T.values.reshape(-1, *image_shape).astype(np.uint8)
        with open(self.data_dir / "annotated_labels.json", "r") as f:
            self.annotated_labels = json.load(f)

        # Chargement des images
        self.X_train = self._load_images(self.data_dir / "X_train")
        self.X_test = self._load_images(self.data_dir / "X_test")

        # Création des index pour supervisé / non-supervisé
        self.supervised_indices = [i for i, mask in enumerate(self.y_train) if mask.sum() > 0]
        self.unsupervised_indices = [i for i, mask in enumerate(self.y_train) if mask.sum() == 0]

    def _load_images(self, dir_path):
        image_paths = sorted(Path(dir_path).glob("*.png"), key=lambda p: int(p.stem))
        if not image_paths:
            raise FileNotFoundError(f"Aucune image trouvée dans {dir_path}")
        images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in image_paths]
        return np.stack(images, axis=0)

    def get_supervised_data(self):
        return self.X_train[self.supervised_indices], self.y_train[self.supervised_indices]

    def get_unsupervised_data(self):
        return self.X_train[self.unsupervised_indices]

    def get_test_data(self):
        return self.X_test