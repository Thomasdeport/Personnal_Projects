import torch
import numpy as np
import pandas as pd

def make_predictions(model, X_test, device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    all_predictions = []  
    all_probabilities = [] 
    
    model.eval()
    with torch.no_grad():
        # Iterate over batches in the test_loader
        for batch_inputs in X_test:
            batch_inputs = batch_inputs[0].to(device,dtype=torch.float32)
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
    df_prediction = pd.DataFrame(data = {'eqt_code_cat':all_predictions})
    df_prediction.to_csv('/kaggle/working/y_predict.csv')
    return df_prediction
