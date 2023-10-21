import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, f1_score
import torch
import torch.nn as nn
import random as rd
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

import json
import csv
import seaborn as sns


#----------------- set seed ------------------
torch.manual_seed(42)
np.random.seed(42)


def hidden_blocks(input_size, output_size, activation_function):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        activation_function,
    )


class MLP(nn.Module):
    def __init__(self, input_size = 75, hidden_units = 512, num_classes = 10, activation_function=nn.LeakyReLU()):
        super(MLP, self).__init__()
        
        self.architecture = nn.Sequential(
            hidden_blocks(input_size, hidden_units, activation_function),
            hidden_blocks(hidden_units, hidden_units, activation_function), 
            hidden_blocks(hidden_units, hidden_units, activation_function),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self,x):
        return self.architecture(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Current device:", device)

    le = LabelEncoder()
    data = pd.read_csv('../akima_v8.50.csv')

    X = data.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
    y = data['skill_id']

    #encode the labels
    y = le.fit_transform(y)

   
    print(le.classes_)
    for i in range(len(le.classes_)):
        print(le.classes_[i], i)
    
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    #splitting and grouping by video_name
    train_idx, test_idx = next(gss.split(X, y, groups=data['video_name']))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    X_train.to_csv('Xtrain_output.csv', index=False)
    y_train_df = pd.DataFrame(y_train, columns=['skill_id'])
    y_train_df.to_csv('Ytrain_output.csv', index=False)

    X_test.to_csv('Xtest_output.csv', index=False)
    y_test_df = pd.DataFrame(y_test, columns=['skill_id'])
    y_test_df.to_csv('Ytest_output.csv', index=False)
    

    # Set the hyperparameters
    input_size = len(data.columns) - 3 # exclude 'id_video', 'frame', 'skill_id'
    hidden_units = 512
    num_classes = len(data['skill_id'].unique())
    lr = 0.0001
    n_epochs = 500
    batch_size = 512


    model = MLP(input_size, hidden_units, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)

    
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device), torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    
     # Load the training data
    '''
    X_train = pd.read_csv('x_train_fold_3.csv')
    y_train = pd.read_csv('y_train_fold_3.csv')
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device), 
                                  torch.LongTensor(y_train['skill_id'].values).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    '''



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
           
            if (i + 1) % int(batch_size/4) == 0:
                acc = 100 * correct / total
                #print the loss and the accuracy for batch_size = 512
                loss_list.append(running_loss / batch_size)
                print('[Epoch %d, Batch %d] Loss: %.3f Accuracy: %.3f%%' % 
                    (epoch + 1, i + 1, running_loss / batch_size, acc))
                

                running_loss = 0.0
                correct = 0
                total = 0
        
        
    # Test the model
    print("Now testing the model...")
    correct = 0
    total = 0

    '''
    # Load the test data
    X_test = pd.read_csv('x_test_fold_3.csv')
    y_test = pd.read_csv('y_test_fold_3.csv')
    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device), 
                                 torch.LongTensor(y_test['skill_id'].values).to(device))
    test_loader = DataLoader(test_dataset, int(batch_size/2), shuffle=False)
    '''    
    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device), torch.LongTensor(y_test).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    y_pred = []
    y_true = []


    with torch.no_grad():
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(test_loader):
            
            #if the input row has a lot of zeros, it is not considered

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            probabilities = torch.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()

            
            gt_labels = labels.tolist()
            raw_predicted = predicted.tolist()
            #check if raw_predicted is an empty list
            
            #create a csv with the first column of ground truth labels and the second column of raw predicted labels
                  
        
            
            '''
            #put them into a csv
            temp = pd.DataFrame({'gt_labels': gt_labels, 'raw_predicted': raw_predicted})

            temp.to_csv('temp_seeing.csv', mode='a', header=False, index=False)
            '''

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss = loss.item()
            acc = 100 * correct / total

            #print all the curve in tensorboard

        
    data_loss_test = {"loss": loss_list}

    #calculate the recall, precision and f1 score

    recall_raw = recall_score(gt_test, raw_pred_test, average='weighted')    
    precision_raw = precision_score(gt_test, raw_pred_test, average='weighted')
    f1_raw = f1_score(gt_test, raw_pred_test, average='weighted')

    data_recall = {"recall": recall_raw}
    data_precision = {"precision": precision_raw}
    data_f1 = {"f1": f1_raw}

    
    

    
    #CONFUSION MATRIX
    #i want a normalized confusion matrix in percentage

    # Create your confusion matrix
    '''
    conf_mat = confusion_matrix(gt_test, raw_pred_test)

    conf_mat_percent = np.around((conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]), decimals=4)

    # Display it with percentage symbols
    f3, ax10 = plt.subplots(1, 1, figsize=(15, 15))
    
    sns.heatmap(conf_mat_percent, annot=True, fmt=".2%", cmap="Blues", ax=ax10, xticklabels=le.classes_, yticklabels=le.classes_)
    '''

    conf_mat = confusion_matrix(gt_test, raw_pred_test)

    conf_mat_percent = np.around((conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]), decimals=4)

    # Display it with percentage symbols
    f3, ax10 = plt.subplots(1, 1, figsize=(15, 15))
    
    sns.heatmap(conf_mat_percent, annot=True, fmt=".4f", cmap="Blues", ax=ax10, xticklabels=le.classes_, yticklabels=le.classes_)


    ax10.set_xlabel('Predicted label')
    ax10.set_ylabel('True label')
    ax10.set_title('Confusion matrix')

    f3.tight_layout()
    f3.savefig('confusion_matrix.png', dpi=300)
    
    data_conf = {"confusion_matrix": conf_mat_percent.tolist()}
    #write to a json file the confusion matrix values with key = 'confusion_matrix'
    
    #ACCURACY ON TEST DATA
    acc_test = 100 * correct / total
    print('Accuracy on test data: %d %%' % (acc_test))
    
    data_acc = {"accuracy": acc_test}
    #save the accuracy on a json file

    data = {
        "recall": recall_raw,
        "precision": precision_raw,
        "f1": f1_raw,
        "loss": loss_list,
        "confusion_matrix": conf_mat_percent.tolist(),
        "accuracy": acc_test,
    }
    

    with open('fold_4.json', 'w') as outfile:
        json.dump(data, outfile)
    
    
    


    np.save('classes.npy', le.classes_)

    torch.save(model.state_dict(), 'model.pth')
    
if __name__ == '__main__':
    main()