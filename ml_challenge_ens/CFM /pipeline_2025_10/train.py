import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, n=10):
    """Affiche les courbes de progression de l'entraînement."""
    length = len(train_losses)
    epochs = range(1, length + 1)

    if length % n == 0:
        plt.clf()

        # Plot de la Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_losses, label="Train Loss", color="blue", linestyle='-', marker='o')
        plt.plot(epochs, val_losses, label="Validation Loss", color="red", linestyle='--', marker='x')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train vs Validation Loss (Epoch {length})')
        plt.legend()

        # Plot de l'Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy", color="blue", linestyle='-', marker='o')
        plt.plot(epochs, val_accuracies, label="Validation Accuracy", color="red", linestyle='--', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Train vs Validation Accuracy (Epoch {length})')
        plt.legend()

        plt.tight_layout()
        plt.show()



def train(
    model,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int = 60,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n: int = 10
):
    """Entraîne le modèle avec CrossEntropyLoss, ReduceLROnPlateau et Early Stopping."""

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    print(f"🔹 Training on device: {device}")

    num_train_samples = len(train_loader.dataset)
    num_val_samples = len(val_loader.dataset)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        with tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for train_batch in pbar:
                train_target = train_batch[1].to(device)
                train_output = model(train_batch[0].to(device))

                loss = criterion(train_output, train_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                prediction = torch.argmax(train_output, dim=1)
                train_accuracy += (prediction == train_target).sum().item() / num_train_samples

                pbar.set_postfix({
                    "Train Loss": f"{train_loss:.4f}",
                    "Train Acc": f"{train_accuracy:.4f}"
                })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_accuracy

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for val_batch in val_loader:
                val_target = val_batch[1].to(device)
                val_output = model(val_batch[0].to(device))

                loss = criterion(val_output, val_target)
                val_loss += loss.item()

                val_accuracy += (torch.argmax(val_output, dim=1) == val_target).sum().item() / num_val_samples

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_accuracy

        print(f"📌 Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"✅ Validation: Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")

        # Scheduler step pour ReduceLROnPlateau
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️ Early stopping patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch}!")
            break

        plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, n=n)

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


