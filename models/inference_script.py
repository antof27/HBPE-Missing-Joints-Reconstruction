import torch
import pandas as pd
import numpy as np

# Load your pre-trained LSTM model
from lstm import LSTMModel

model = LSTMModel()
model.load_state_dict(torch.load('lstm_weight.pth'))
model.eval()

# Load and preprocess your new dataset
new_dataset = pd.read_csv('./original_datasets/dataset_v8.50.csv')

# Apply the same preprocessing as during training for each column
# For example, normalize the data for each column
# ...

# Replace missing values with a placeholder for each column
placeholder_value = 0.0
for column in new_dataset.columns:
    new_dataset[column] = new_dataset[column].replace(0.0, placeholder_value)

# Define a function to create sequences
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return sequences

# Define the sequence length
sequence_length = 10  # Adjust this value as needed

# Create sequences for prediction for each column
sequences = {}
for column in new_dataset.columns:
    sequences[column] = create_sequences(new_dataset[column].values, sequence_length)

predictions = {}
with torch.no_grad():
    for column in new_dataset.columns:
        predictions[column] = []
        for i in range(len(sequences[column]) - sequence_length + 1):
            # Extract a sequence of length sequence_length
            input_sequence = sequences[column][i:i + sequence_length]
            
            # Flatten the sequence and ensure the input_sequence has the shape [sequence_length, 1, 75]
            input_sequence = torch.tensor(input_sequence).float().view(sequence_length, 1, 75)
            
            prediction = model(input_sequence)
            predictions[column].append(prediction.item())




# Interpolate the missing values for each column
for column in new_dataset.columns:
    new_dataset[column] = new_dataset[column].replace(placeholder_value, predictions[column])

# Save the interpolated dataset
new_dataset.to_csv('interpolated_dataset.csv', index=False)
