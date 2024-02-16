#i have a dataset, I want to count the number of zeros in the dataset 

import pandas as pd
import numpy as np
import os 
# Get the current script's file path
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)
g_parent_directory = os.path.dirname(parent_directory)
data_directory = g_parent_directory + '/data/'

# Load the dataset
dataset = pd.read_csv(data_directory + "zero_filled_latest.csv")

#count all the zero values in the dataset

zero_values = dataset.isin([0]).sum().sum()

print("The number of zero values in the dataset is: ", zero_values)