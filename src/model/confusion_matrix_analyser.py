import json
import matplotlib.pyplot as plt

# Load the confusion matrix from the final_results.json file
with open('./zero_datasets/results/final_results.json') as f:
    data = json.load(f)
    confusion_matrix = data['confusion_matrix']
    print(confusion_matrix)
    print(len(confusion_matrix))

#for each class, calculate the f1 score 
f1_score = []
precision = []
recall = []
for i in range(len(confusion_matrix)):
    tp = confusion_matrix[i][i]
    fp = 0
    fn = 0
    for j in range(len(confusion_matrix)):
        if j != i:
            fp += confusion_matrix[j][i]
            fn += confusion_matrix[i][j]
    f1_score.append(2*tp/(2*tp+fp+fn))
    precision.append(tp/(tp+fp))
    recall.append(tp/(tp+fn))
    
    

print("F1 SCORE : ")
print(f1_score)
print("PRECISION : ")
print(precision)
print("RECALL : ")
print(recall)






