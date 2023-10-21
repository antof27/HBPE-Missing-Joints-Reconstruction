import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import torch
from torch import nn, optim
from sklearn.preprocessing import LabelEncoder
import json


# ----------------- Set Seed ------------------
torch.manual_seed(42)
np.random.seed(42)


def hidden_blocks(input_size, output_size, activation_function):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        activation_function,
    )


class MLP(nn.Module):
    def __init__(self, input_size=75, hidden_units=512, num_classes=10, activation_function=nn.LeakyReLU()):
        super(MLP, self).__init__()

        self.architecture = nn.Sequential(
            hidden_blocks(input_size, hidden_units, activation_function),
            hidden_blocks(hidden_units, hidden_units, activation_function),
            hidden_blocks(hidden_units, hidden_units, activation_function),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        return self.architecture(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Load the training data
    path = './nn_datasets'
    fold = 1

    X_train = pd.read_csv(path + '/x_train_fold_' + str(fold) + '.csv')
    y_train = pd.read_csv(path + '/y_train_fold_' + str(fold) + '.csv')

    input_size = X_train.shape[1]  # Assuming X_train contains your data

    # Set the hyperparameters
    hidden_units = 512
    num_classes = len(y_train['skill_id'].unique())
    lr = 0.0001
    n_epochs = 500
    batch_size = 512

    model = MLP(input_size, hidden_units, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device),
                                  torch.LongTensor(y_train['skill_id'].values).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    raw_pred_test = []
    loss_list = []
    gt_test = []

    print("Now training the model...")
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            if (i + 1) % int(batch_size / 4) == 0:
                acc = 100 * correct / total
                # Print the loss and the accuracy for batch_size = 512
                loss_list.append(running_loss / batch_size)
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / batch_size:.3f} Accuracy: {acc:.3f}%')

                running_loss = 0.0
                correct = 0
                total = 0

    # Test the model
    print("Now testing the model...")
    correct = 0
    total = 0

    # Load the test data
    X_test = pd.read_csv(path + '/x_test_fold_' + str(fold) + '.csv')
    y_test = pd.read_csv(path + '/y_test_fold_' + str(fold) + '.csv')

    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device),
                                 torch.LongTensor(y_test['skill_id'].values).to(device))
    test_loader = DataLoader(test_dataset, int(batch_size / 2), shuffle=False)

    with torch.no_grad():
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(test_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            probabilities = torch.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()

            gt_labels = labels.tolist()
            raw_predicted = predicted.tolist()

            gt_test.extend(gt_labels)
            raw_pred_test.extend(raw_predicted)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss = loss.item()
            acc = 100 * correct / total

    # Calculate the recall, precision, and f1 score
    recall_raw = recall_score(gt_test, raw_pred_test, average='weighted')
    precision_raw = precision_score(gt_test, raw_pred_test, average='weighted')
    f1_raw = f1_score(gt_test, raw_pred_test, average='weighted')

    conf_mat = confusion_matrix(gt_test, raw_pred_test)
    conf_mat_percent = np.around((conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]), decimals=4)

    # Accuracy on test data
    acc_test = 100 * correct / total
    print(f'Accuracy on test data: {acc_test:.2f}%')

    # Save the accuracy in a JSON file
    data = {
        "recall": recall_raw,
        "precision": precision_raw,
        "f1": f1_raw,
        "loss": loss_list,
        "confusion_matrix": conf_mat_percent.tolist(),
        "accuracy": acc_test,
    }

    with open(path + f'/results/results_fold_{fold}.json', 'w') as outfile:
        json.dump(data, outfile)

    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
