"""
results_elaborator.py

This module processes and analyzes results data from JSON files
and generates summary statistics and visualizations.

"""

import os
import json
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)

# Initialize a list to store the data from each JSON file
temp_recall = []
temp_precision = []
temp_f1 = []
temp_loss_list = []
temp_conf_matrix = []
temp_accuracy = []
mean_recall = 0
mean_precision = 0
mean_f1 = 0
mean_loss_list = []
mean_conf_matrix = []
mean_accuracy = 0

environment = os.getcwd()

fold = 'linear_datasets/results/'
environment = os.path.join(environment, fold)

# Iterate on the environment folder
for files in os.listdir(environment):
    file_path = os.path.join(environment, files)

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Get the data from the JSON file
    temp_recall.append(data["recall"])
    temp_precision.append(data["precision"])
    temp_f1.append(data["f1"])
    temp_loss_list.append(data["loss"])
    temp_conf_matrix.append(data["confusion_matrix"])
    temp_accuracy.append(data["accuracy"])

# Calculate the mean of the data values
mean_recall = np.mean(temp_recall)
mean_precision = np.mean(temp_precision)
mean_f1 = np.mean(temp_f1)
mean_loss_list = np.mean(temp_loss_list, axis=0)
mean_conf_matrix = np.mean(temp_conf_matrix, axis=0)
mean_accuracy = np.mean(temp_accuracy)

# Save the results in a JSON file
results = {
    "recall": mean_recall,
    "precision": mean_precision,
    "f1": mean_f1,
    "loss_list": mean_loss_list.tolist(),
    "confusion_matrix": mean_conf_matrix.tolist(),
    "accuracy": mean_accuracy,
}

with open("./" + fold + "final_results.json", "w") as f:
    json.dump(results, f)

# Create a plot for the loss
f1, ax8 = plt.subplots(1, 1, figsize=(10, 6))
x3 = np.linspace(1, len(mean_loss_list), len(mean_loss_list))
y3 = mean_loss_list
ax8.plot(x3, y3)
ax8.set_xlabel('Epochs')
ax8.set_ylabel('Loss')
f1.tight_layout()
f1.savefig('loss.png', dpi=300)

# --------------------------CONFUSION MATRIX-------------------------------

conf_mat = mean_conf_matrix * 100
conf_mat_percent = np.around((conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]), decimals=4)

# Display it with percentage symbols
f3, ax10 = plt.subplots(1, 1, figsize=(15, 15))

sns.heatmap(conf_mat_percent, annot=True, fmt=".2",
            cmap="Blues", ax=ax10, xticklabels=le.classes_, yticklabels=le.classes_)
'''
# Display it
f3, ax10 = plt.subplots(1, 1, figsize=(10, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_mat_percent,
    display_labels=le.classes_,
)

disp.plot(cmap=plt.cm.Blues, ax=ax10, colorbar=True, xticks_rotation=90)
'''
ax10.set_xlabel('Predicted label')
ax10.set_ylabel('True label')
ax10.set_title('Confusion matrix')

f3.tight_layout()
f3.savefig('confusion_matrix.png', dpi=300)

# --------------------------ACCURACY-------------------------------

print("Accuracy: ", mean_accuracy)
print("Recall: ", mean_recall)
print("Precision: ", mean_precision)
print("F1: ", mean_f1)
