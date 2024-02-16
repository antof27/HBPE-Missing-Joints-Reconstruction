import csv
import pandas as pd 
from matplotlib import pyplot as plt    
import os 

# Get the current script's file path
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)
g_parent_directory = os.path.dirname(parent_directory)
data_directory = g_parent_directory + '/data/'

df = pd.read_csv(data_directory + 'dataset_v8.50.csv')


skills_occurrences = df['skill_id'].value_counts()  
print("Skills occurrences: ", skills_occurrences)

# Plot the skills occurrences in a pie chart
labels = skills_occurrences.index.tolist()
labels = [str(i) for i in labels]
labels = [i.upper() for i in labels]
plt.pie(skills_occurrences, labels=labels, autopct='%1.1f%%')
plt.title('Class percentages')
plt.savefig('skills_occurrences.png', dpi=300)

