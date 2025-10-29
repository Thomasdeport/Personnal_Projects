import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# ==========================================================
# Plot training curves
# ==========================================================
def plot_training_curves(history):
    """Plot training & validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))

    # Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="red")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", color="blue")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ==========================================================
# Training loop (multi-GPU + mixed precision ready)
# ==========================================================
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    criterion=None,
    num_epochs: int = 50,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_plots: bool = True,
    save_best: bool = True,
    save_path: str = "best_model.pt",
    use_amp: bool = True,
):
    """
    Training loop with tqdm progress, mixed precision, early stopping,
    LR scheduling, and auto multi-GPU support.

    Args:
        model: torch.nn.Module
        train_loader: training DataLoader
        val_loader: validation DataLoader
        optimizer: optimizer instance
        scheduler: learning rate scheduler (optional)
        criterion: loss function (default: CrossEntropyLoss)
        num_epochs: number of epochs
        patience: early stopping patience
        device: cuda or cpu
        show_plots: display learning curves at end
        save_best: save best model checkpoint
        save_path: path to save best model
        use_amp: use mixed precision training
    """

    # --- Default loss ---
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    # --- Multi-GPU handling ---
    if torch.cuda.device_count() > 1 and device == "cuda":
        if not isinstance(model, torch.nn.DataParallel):
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)

    model = model.to(device)

    # --- Mixed precision tools ---
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Tracking ---
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    start_time = time.time()

    print(f"\n Training on {device} ({'AMP ON' if use_amp else 'AMP OFF'})\n")

    # ==========================================================
    # Main training loop
    # ==========================================================
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_total += targets.size(0)

            pbar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

        # ---- End of epoch: compute metrics ----
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        # ======================================================
        # Validation
        # ======================================================
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == targets).sum().item()
                val_total += targets.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total

        # ======================================================
        # Log metrics
        # ======================================================
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}"
        )

        # ======================================================
        # Scheduler step (ReduceLROnPlateau or standard)
        # ======================================================
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # ======================================================
        # Early stopping
        # ======================================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            patience_counter = 0
            if save_best:
                torch.save(best_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # ======================================================
        # Live plotting every few epochs
        # ======================================================
        if show_plots and epoch % max(1, num_epochs // 2) == 0:
            plot_training_curves(history)

    # ==========================================================
    # Restore best weights
    # ==========================================================
    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed/60:.2f} min | Best Val Loss: {best_val_loss:.4f}\n")

    if show_plots:
        plot_training_curves(history)

    return model, history

