
import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

#STANDARD PARAMETERS#

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = .... 
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


#LOSSES DEFINITIONS#

#MCC LOSS 

def plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, n=10):
    length = len(train_losses)
    epochs = range(1, length + 1)

    if length % n == 0:
        fig = plt.gcf()  # Get the current figure
        if fig is None:
            fig = plt.figure(figsize=(10, 6))

        plt.clf()  # Clear the current figure

        # Loss Plot
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_losses, label="Train Loss", color="blue", linestyle='-', marker='o')
        plt.plot(epochs, val_losses, label="Validation Loss", color="red", linestyle='--', marker='x')
        plt.xlabel('Epoch')

        if (max(train_losses)-min(train_losses))/(max(train_losses)+min(train_losses))>1/10:
            plt.yscale('log')
            plt.ylabel('Loss (Logscale)')
        else:
            plt.ylabel('Loss')
        plt.title(f'Train vs Validation Loss, epoch = {length}')
        plt.legend()

        # Accuracy Plot
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy", color="blue", linestyle='-', marker='o')
        plt.plot(epochs, val_accuracies, label="Validation Accuracy", color="red", linestyle='--', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Train vs Validation Accuracy, epoch = {length}')
        plt.legend()

        # Afficher la figure
        plt.tight_layout()
        plt.draw()
        plt.show()


class MCCLoss(nn.Module):
    def __init__(self, T=2.5):
        super(MCCLoss, self).__init__()
        self.T = T

    def entropy(self, input):
        """Calcule l'entropie de chaque élément du batch"""
        input = torch.clamp(input, min=1e-5)  # Éviter log(0)
        return -torch.sum(input * torch.log(input), dim=1)

    def forward(self, targets):
        """
        Calcul de la MCC Loss
        - targets: tenseur de forme (batch_size, nb_class) contenant les logits du modèle
        """
        batch_size, nb_class = targets.size()

        # Appliquer la Softmax avec la température T
        rescaled_targets = F.softmax(targets / self.T, dim=1)

        # Calcul des poids en fonction de l'entropie
        targets_weights = self.entropy(rescaled_targets).detach()
        targets_weights = 1 + torch.tanh(-targets_weights)
        targets_weights = batch_size * targets_weights / torch.sum(targets_weights)

        # Matrice de covariance sans normalisation excessive
        cov_matrix = (
            rescaled_targets.mul(targets_weights.view(-1, 1))
            .transpose(1, 0)
            .mm(rescaled_targets)
        )
        cov_matrix /= torch.sum(cov_matrix)  # Normalisation simple

        # Calcul de la MCC Loss
        loss = (torch.sum(cov_matrix) - torch.trace(cov_matrix)) / nb_class

        # le rapport entre la cross entropy et la MCC loss est d'environ 50- 100 par batch on multiplie par un
        # coeff pour rendre cette fonction perte significative
        return 50*loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, outputs, targets):
        targets = targets.long()  # S'assurer que c'est bien un entier
        num_classes = outputs.size(1)  # Nombre de classes

        # Convertir en one-hot encoding
        true_dist = torch.zeros_like(outputs).scatter_(1, targets, 1)

        # Appliquer le lissage
        true_dist = true_dist * (1 - self.smoothing) + self.smoothing / num_classes

        # Calcul de la perte
        log_probs = nn.functional.log_softmax(outputs, dim=-1)
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        
        # scaling en fonction du ratio par rappport à la cross entropy loss 
        return loss/4


def train(
    model,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int = 60,
    alpha: float = 0.1,
    beta: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    test_every: int = 10,
    n:int = 10 
):
    """Trains the model with Label Smoothing and MCC Loss."""

    # Define loss functions
    criterion = nn.CrossEntropyLoss()
    label_smoothing_loss_fn = LabelSmoothingLoss()
    mcc_loss_fn = MCCLoss()

    # Send the model to the target device (CPU/GPU)
    model.to(device)
    print(f"🔹 Training on device: {device}")

    best_val_accuracy = 0.0
    best_model_state = None

    # Track metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_accuracy, total_samples = 0.0, 0.0, 0
        mcc_loss_accumulated = 0.0  # Track MCC Loss
        label_smooting_loss_acc = 0.0
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            label_smooting_loss_acc += alpha * label_smoothing_loss_fn(outputs, targets)
            loss = criterion(outputs, targets.argmax(dim=1)) + alpha * label_smoothing_loss_fn(outputs, targets)

            # MCC Loss (added periodically)
            if batch_idx % test_every == 0:
                mcc_loss = mcc_loss_fn(outputs)
                mcc_loss_accumulated += beta * mcc_loss.item()
                loss += beta * mcc_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            targets_accs = targets.argmax(dim=1)
            train_accuracy += (predicted == targets_accs).sum().item()
            total_samples += targets.size(0)

        # Update scheduler
        scheduler.step()
        # Compute average metrics
        train_loss /= len(train_loader)
        train_accuracy /= total_samples
        mcc_loss_accumulated /= max(1, len(train_loader) // test_every)
        label_smooting_loss_acc /= len(train_loader)
        loss += beta * mcc_loss_accumulated
        
        print(f"📌 Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, MCC Loss: {mcc_loss_accumulated:.4f}, Label Smoothing: {label_smooting_loss_acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_accuracy, val_samples = 0.0, 0.0, 0
        mcc_loss_accumulated, label_smooting_loss_acc = 0, 0 
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                label_smooting_loss_acc += alpha * label_smoothing_loss_fn(outputs, targets)
                loss = criterion(outputs, targets.argmax(dim=1)) + alpha * label_smoothing_loss_fn(outputs, targets)
                
                # MCC Loss (added periodically)
                if batch_idx % test_every == 0:
                    mcc_loss = mcc_loss_fn(outputs)
                    mcc_loss_accumulated += beta * mcc_loss.item()
                    loss += beta * mcc_loss

                val_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                targets_accs = targets.argmax(dim=1)
                val_accuracy += (predicted == targets_accs).sum().item()
                val_samples += targets.size(0)

        val_loss /= len(val_loader)
        val_accuracy /= val_samples
        mcc_loss_accumulated /= max(1, len(val_loader) // test_every)
        label_smooting_loss_acc /= len(val_loader)

        print(f"✅ Validation: Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, MCC Loss: {mcc_loss_accumulated:.4f}, Label Smoothing: {label_smooting_loss_acc:.4f}")

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Update best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
        plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, n=n)
        
        # Print parameter gradients every 'n' epochs
        if epoch % n == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {param.grad.abs().mean():.6f}")

    # Load the best model
    model.load_state_dict(best_model_state)

    return model
