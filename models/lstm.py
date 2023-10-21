import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class LSTMModel(nn.Module):
    def __init__(self, input_size=75, hidden_size=5, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Check if a GPU is available
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your data from a CSV file
    data = pd.read_csv('./original_datasets/selected_sequences_5.csv')

    # Define the probability of setting a value to zero
    probability = 0.2  # Adjust this to control the percentage of values set to zero

    # Generate a random mask with the same shape as the dataset
    mask = np.random.choice([0, 1], size= data.shape, p=[probability, 1 - probability])

    # Apply the mask to set some values to zero
    masked_data = data * mask

    # Now, 'masked_data' contains your original data with some values set to zero.

    # Proceed with your previous code to replace non-zero values with NaN
    n_timesteps = 1
    filled_data = pd.DataFrame()

    columns_to_include = masked_data.columns[:-3]

    for col in columns_to_include:
        sequences = [masked_data.iloc[i:i + n_timesteps][col].values for i in range(len(masked_data) - n_timesteps + 1)]
        filled_data[col] = np.concatenate(sequences, axis=0)

    filled_data = filled_data.apply(lambda x: x.replace(0.0, np.nan))

    #export the filled data to a new csv file
    filled_data.to_csv('missing_data.csv', index=False)


    # Convert data to PyTorch tensors and move them to the GPU
    X = torch.tensor(filled_data.values, dtype=torch.float32).to(device)
    y = torch.tensor(filled_data.values[:, 1], dtype=torch.float32).view(-1, 1).to(device)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_size = len(columns_to_include)
    hidden_size = 5
    output_size = 1

    model = LSTMModel(input_size, hidden_size, output_size).to(device)

    # Define loss and optimizer on the GPU
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters())

    # Create a DataLoader for batch training
    batch_size = 1
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    num_epochs = 20
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.view(-1, n_timesteps, input_size))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate model on the test data
    X_test = X_test.view(-1, n_timesteps, input_size)
    yhat_test = model(X_test).cpu()

    # Update the original data with predicted values
    for col in columns_to_include:
        original_col = data[col].values
        missing_values = np.isnan(original_col)

        # Find the indices where missing values are True and assign predicted values
        missing_indices = np.where(missing_values)[0]
        original_col[missing_indices] = yhat_test[missing_indices].detach().numpy().flatten()

        # Update the original data with predicted values
        data[col] = original_col

    
    # Save the state_dict of the model
    torch.save(model.state_dict(), 'lstm_weight.pth')


    # Export the updated DataFrame to a new CSV file, including predicted values
    data.to_csv('updated_data.csv', index=False)


if __name__ == '__main__':
    main()