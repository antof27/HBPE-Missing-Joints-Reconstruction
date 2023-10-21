import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import random as rd
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

#----------------- set seed ------------------
torch.manual_seed(42)
np.random.seed(42)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Current device:", device)

    le = LabelEncoder()
    '''
    print(le.classes_)
    for i in range(len(le.classes_)):
        print(le.classes_[i], i)
    '''
    
    folder = "zero_datasets"
    data = pd.read_csv(f'./{folder}/zero_v8.50.csv')
    
    

    #X = data.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
    X = data
    
    y = data['skill_id']

    #encode the labels
    y = le.fit_transform(y)
 
    
    n_folds = 5

# Define the k-fold cross-validation object
    gkf = GroupKFold(n_splits=n_folds)

# Loop over the folds
    for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups=data['video_name'])):
        # Split the data into training and test sets for this fold
        X_train, y_train = X.iloc[train_index], y[train_index], 
        X_test, y_test = X.iloc[test_index], y[test_index]

        #save the dataset with its first 75 column 
        X_train.iloc[:, :75].to_csv(f'./{folder}/x_train_fold_{fold}.csv', index=False)
        pd.DataFrame(y_train, columns=['skill_id']).to_csv(f'./{folder}/y_train_fold_{fold}.csv', index=False)
        X_test.iloc[:, :75].to_csv(f'./{folder}/x_test_fold_{fold}.csv', index=False)
        pd.DataFrame(y_test, columns=['skill_id']).to_csv(f'./{folder}/y_test_fold_{fold}.csv', index=False)
        
if __name__ == '__main__':
    main()

