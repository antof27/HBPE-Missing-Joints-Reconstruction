import csv
import os

# Get the current script's file path
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)
g_parent_directory = os.path.dirname(parent_directory)
data_directory = g_parent_directory + '/data'


# Open the original CSV file for reading
with open('dataset_v8.50.csv', 'r') as f:
    reader = csv.reader(f)
    
    # Read and discard the header row
    header = next(reader)

    # Initialize variables for tracking sequences and rows
    sequences = []
    current_sequence = []

    previous_frame = None  # Initialize previous_frame
    previous_video = None  # Initialize previous_video

    for row in reader:
        # Check if all values in the first 75 columns are different from zero
        if all(float(val) != 0.0 for val in row[:75]):
            current_frame = int(row[76])  # Convert frame value to an integer
            current_video = row[75]  # Get the current video

            if previous_frame is not None:
                # Calculate the difference between the current frame and the previous frame
                frame_difference = current_frame - previous_frame
                
                # Check if the frame difference is less than 2
                if frame_difference < 2 and current_video == previous_video:
                    current_sequence.append(row)
                else:
                    if len(current_sequence) >= 5:
                        sequences.extend(current_sequence)
                    current_sequence = []

            # Set the current frame and video as the previous frame and video for the next iteration
            previous_frame = current_frame
            previous_video = current_video

    if len(current_sequence) >= 5:
        sequences.extend(current_sequence)

# Write the selected sequences to a new CSV file
with open('selected_sequences_5.csv', 'w', newline='') as new_file:
    writer = csv.writer(new_file)
    
    # Write the header row
    writer.writerow(header)
    
    # Write the selected sequences
    writer.writerows(sequences)

print("Selected sequences have been saved to 'selected_sequences_5.csv'.")

with open('selected_sequences_5.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    row_count = len(data)
    print(row_count)
