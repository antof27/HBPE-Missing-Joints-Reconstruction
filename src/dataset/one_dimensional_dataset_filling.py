import pandas as pd
import os
import sys

# Get the current script's file path
script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)
interpolations_directory = parent_directory + '/interpolations'
sys.path.insert(1, interpolations_directory)
g_parent_directory = os.path.dirname(parent_directory)
data_directory = g_parent_directory + '/data'

from akima_interpolation import akima_interpolation
from spline_interpolation import spline_interpolation
from inverse_distance import inverse_distance_weighting_with_missing
from pchip_interpolation import pchip_interpolation
from interp1d import interp1d_interpolation


# Read the original CSV file
df = pd.read_csv(data_directory + '/zero_filled_latest.csv')
header = df.columns.values.tolist()

# Initialize variables
current_sequence = []
column_interpolated = []
interpolated_df = pd.DataFrame()
threshold = 2
method = spline_interpolation
# Iterate through the columns
for i in range(0, len(header)):
    column_interpolated = []
    current_sequence = []   
    # Iterate through the rows
    print("Processing column " + str(i) + " of " + str(len(header)))
    if i >= 75:
        column_interpolated = df.iloc[:, i].tolist()
        interpolated_df[header[i]] = column_interpolated
        continue

    for j in range(0, len(df) - 1):  # Skip the last row to avoid an out-of-bounds error
        current_frame = df.iloc[j, 76]
        next_frame = df.iloc[j + 1, 76]
        current_video = df.iloc[j, 75]
        next_video = df.iloc[j + 1, 75]
        
        if next_frame - current_frame < 2 and current_video == next_video:
            current_sequence.append(df.iloc[j, i])
        else:
            current_sequence.append(df.iloc[j, i])
            interpolated_sequence = method(current_sequence)
            column_interpolated.extend(interpolated_sequence)
            current_sequence = []

    # Include the last row's value
    current_sequence.append(df.iloc[len(df) - 1, i])
    interpolated_sequence = method(current_sequence)
    column_interpolated.extend(interpolated_sequence)

    print("len(column_interpolated): " + str(len(column_interpolated)))


    interpolated_df[header[i]] = column_interpolated

# Save the interpolated data to a new CSV file
interpolated_df.to_csv(data_directory + '/spline_interpolated.csv', index=False)




 
