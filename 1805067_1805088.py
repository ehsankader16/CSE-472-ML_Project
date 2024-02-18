# These dependencies are necessary for loading the data
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import soundfile as sf
import openmic.vggish
from IPython.display import Audio

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Be sure to set this after downloading the dataset!
DATA_ROOT = './'

OPENMIC = np.load(os.path.join(DATA_ROOT, 'openmic-2018.npz'), allow_pickle=True)

print(list(OPENMIC.keys()))

X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

with open(os.path.join(DATA_ROOT, 'class-map.json'), 'r') as f:
    class_map = json.load(f)
    
split_train = pd.read_csv(os.path.join(DATA_ROOT, 'partitions/split01_train.csv'), 
                          header=None)
split_test = pd.read_csv(os.path.join(DATA_ROOT, 'partitions/split01_test.csv'), 
                         header=None)

train_set = set(split_train[0])
test_set = set(split_test[0])

idx_train, idx_test = [], []

for idx, n in enumerate(sample_key):
    if n in train_set:
        idx_train.append(idx)
    elif n in test_set:
        idx_test.append(idx)
    else:
        # This should never happen, but better safe than sorry.
        raise RuntimeError('Unknown sample key={}! Abort!'.format(sample_key[idx]))
        
# Finally, cast the idx_* arrays to numpy structures
idx_train = np.asarray(idx_train)
idx_test = np.asarray(idx_test)

X_train = X[idx_train]
X_test = X[idx_test]

Y_true_train = Y_true[idx_train]
Y_true_test = Y_true[idx_test]

Y_mask_train = Y_mask[idx_train]
Y_mask_test = Y_mask[idx_test]
    
torch.autograd.set_detect_anomaly(True)

class InstrumentClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(InstrumentClassifier, self).__init__()
        # self.conv1d = nn.Conv1d(in_channels=input_features, out_channels=128, kernel_size=3, stride=1, padding=1)  # Adjust the parameters as necessary
        # self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)  # Adjust hidden_size as necessary
        # self.dense = nn.Linear(64, num_classes)  # Final layer for multi-label classification
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=input_features, stride=1)  # Adjust the parameters as necessary
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)  # Adjust hidden_size as necessary
        self.dense = nn.Linear(64, num_classes)  # Final layer for multi-label classification

    def forward(self, x):
        # Assuming input x is of shape (batch, channels, sequence_length)
        # x = F.relu(self.conv1d(x))
        # x, (h_n, c_n) = self.lstm(x)
        # x = x[:, -1, :]  # Use the output of the last LSTM time step
        # x = torch.sigmoid(self.dense(x))  # Use sigmoid for multi-label classification
        # return x
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1d(x))
        x = x.transpose(1, 2)  # Adjust dimensions for LSTM
        x, (h_n, c_n) = self.lstm(x)
        x = x[:, -1, :]
        x = torch.sigmoid(self.dense(x))
        return x
    
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()  # Apply threshold to get binary predictions
            all_predictions.extend(predictions.numpy())
            all_targets.extend(targets.numpy())

    # Convert lists to numpy arrays for sklearn metrics
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0).astype(int)
    all_targets = np.concatenate(all_targets, axis=0).astype(int)

    # Calculate metrics
    precision = precision_score(all_targets, all_predictions, average='micro')
    recall = recall_score(all_targets, all_predictions, average='micro')
    f1 = f1_score(all_targets, all_predictions, average='micro')
    accuracy = accuracy_score(all_targets.flatten(), all_predictions.flatten())

    return precision, recall, f1, accuracy

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_true_train_tensor = torch.tensor(Y_true_train, dtype=torch.float32)  # Ensure Y_true_train is suitable for multi-label classification
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_true_test_tensor = torch.tensor(Y_true_test, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, Y_true_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_true_test_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Adjust batch_size as necessary
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize the model
# num_features = X_train.shape[1]  # Adjust based on your input feature size
# num_classes = len(class_map)  # Number of classes for multi-label classification
# model = InstrumentClassifier(input_features=num_features, num_classes=num_classes)
num_features = X_train.shape[2]  # Adjust based on your input feature size
num_classes = Y_true_train.shape[1]  # Number of classes for multi-label classification
model = InstrumentClassifier(input_features=num_features, num_classes=num_classes)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adjust learning rate as necessary

num_epochs = 100
learning_rate = 0.01

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        try:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or INF loss detected at epoch {epoch+1}, batch {i+1}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # with torch.no_grad():
            #     for param in model.parameters():
            #         if param.grad is not None:
            #             param -= learning_rate * param.grad
                
        except Exception as e:
            print(f"Error occurred at epoch {epoch+1}, batch {i+1}: {e}")
            break
        
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    
precision, recall, f1, accuracy = evaluate_model(model, test_loader)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}')