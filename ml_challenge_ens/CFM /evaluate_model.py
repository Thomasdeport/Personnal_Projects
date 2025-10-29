import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Évaluation du modèle sur le jeu de validation ---

def evaluate_model(model, val_loader, device, traintest = "train"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Convertir en arrays numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Calcul des métriques principales ---
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification Report :")
    print(classification_report(all_labels, all_preds, digits=4))

    # --- Matrice de confusion ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.title(f"Matrice de confusion - {traintest} set")
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables classes")
    plt.show()

    return acc, cm
