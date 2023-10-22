import numpy as np
import pandas as pd
import random 

# Read the dataset from a CSV file
df = pd.read_csv('./original_datasets/selected_sequences_5.csv')

# Define parameters
max_sequence_length = 5
step = 3

#i need to iterate all the rows and all the columns with a step equal to 3

# Iterate through the columns

for i in range(0, len(df.columns)-3, step):
    # Iterate through the rows
    print("column: ", i)
    for j in range(1, len(df) - 1):  # Skip the last row to avoid an out-of-bounds error

        current_frame = df.iloc[j, 76]
        current_video = df.iloc[j, 75]

        
        previous_frame = df.iloc[j - 1, 76]
        previous_video = df.iloc[j - 1, 75]

        if j < len(df) - 1:
            next_frame = df.iloc[j + 1, 76]
            next_video = df.iloc[j + 1, 75]


        #if next_frame - current_frame < 2 and current_video == next_video:
        
        if j < 3 or j >= len(df) - 1 or current_video != previous_video or current_video != next_video or current_frame - previous_frame > 1 or next_frame - current_frame > 1: 
            
            continue
        else:
            
            if random.random() < 0.7:
            #define a random sequence of numbers to be set to 0
                sequence_length = np.random.randint(1, max_sequence_length)
            #define the index where the sequence will start
            
                index = np.random.randint(2, len(df) - sequence_length)

                while df.iloc[index, 75] != df.iloc[index-1, 75] or df.iloc[index, 75] != df.iloc[index+1, 75] or df.iloc[index, 76] - df.iloc[index-1, 76] > 1 or df.iloc[index+1, 76] - df.iloc[index, 76] > 1:
                    index = np.random.randint(2, len(df) - sequence_length)
                    

                
            #set the sequence to 0 and the corrisponding values of the next 2 columns
                #count how many elements are next until the next frame
                
                count = 0
                while index+count < len(df)-1 and df.iloc[index, 75] == df.iloc[index+count, 75] and df.iloc[index+count, 76] - df.iloc[index, 76] <= count:
                    count += 1
                
                sequence_length = min(sequence_length, count-1)

                df.iloc[index:index + sequence_length, i:i + step] = 0

                j += sequence_length
    


# Print the modified DataFrame
print(df)

# Save the modified DataFrame to a new CSV file
df.to_csv('zero_filled_7.1.csv', index=False)
