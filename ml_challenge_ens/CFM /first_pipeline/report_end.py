import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd



def make_predictions(model, X_test, batch_size=256, device = "cuda" if torch.cuda.is_available() else "cpu"):
    X_sub = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_loader = DataLoader(X_sub, batch_size=batch_size, shuffle=False)
    
    all_predictions = []  
    all_probabilities = [] 
    
    model.eval()
    with torch.no_grad():
        # Iterate over batches in the test_loader
        for batch_inputs in test_loader:
            batch_inputs = batch_inputs.to(device,dtype=torch.float32)  # Move inputs to the correct device (GPU/CPU)

            # Make predictions for the current batch
            outputs = model(batch_inputs)
            probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to get class probabilities
            predicted_classes = torch.argmax(probabilities, dim=1)  # Get the class with the highest probability

            # Append predictions and probabilities to their respective lists
            all_predictions.append(predicted_classes.cpu().numpy())  # Move to CPU and convert to numpy
            all_probabilities.append(probabilities.cpu().numpy())  # Move to CPU and convert to numpy
    
    # Concatenate results from all batches
    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    
    # Print the predictions for the entire dataset
    print("Predictions for the entire dataset:", all_predictions)
    df_prediction = pd.DataFrame(data = {'eqt_code_cat':all_predictions})

    return df_prediction
