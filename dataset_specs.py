import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("./original_datasets/zero_filled_7.csv")


# Create a list to store the lengths of zero sequences
zero_sequence_lengths = []

# Iterate through each column
for col in df.columns:
    # Iterate through each row in the column
    zero_count = 0
    for value in df[col]:
        if value == 0.0:
            zero_count += 1
        elif zero_count > 0:
            zero_sequence_lengths.append(zero_count)
            zero_count = 0
    # Check if the last value in the column is zero
    if zero_count > 0:
        zero_sequence_lengths.append(zero_count)

# Calculate the maximum sequence length
max_sequence_length = max(zero_sequence_lengths)

# Plot a histogram of zero sequence lengths with one bin per integer
plt.hist(zero_sequence_lengths, bins=range(1, max_sequence_length + 2), edgecolor='k')
plt.title('Histogram of Zero Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.xticks(range(1, 50))
plt.xlim(1, 50)
plt.show()







'''
import pandas as pd

def analyze_dataset(csv_file_path):
    # Read the dataset from the CSV file using pandas
    df = pd.read_csv(csv_file_path)
    
    # Total number of rows and columns
    total_rows, total_columns = df.shape
    
    # Total number of values equal to 0.0
    total_zeros = (df == 0.0).sum().sum()
    
    # Total number of known (non-zero) values
    total_known_values = (df != 0.0).sum().sum()
    
    # Count the total number of sequences of zeros along all rows of all columns
    zero_sequences = 0
    in_sequence = False
    for index, row in df.iterrows():
        if all(row == 0.0):
            if not in_sequence:
                zero_sequences += 1
                in_sequence = True
        else:
            in_sequence = False

    return {
        "Total Rows": total_rows,
        "Total Columns": total_columns,
        "Total Zeros": total_zeros,
        "Total Known Values": total_known_values,
        "Total Zero Sequences": zero_sequences
    }

# Example usage:
csv_file_path = "./original_datasets/zero_filled_7.csv"
results = analyze_dataset(csv_file_path)
print("Dataset Specifications:")
for key, value in results.items():
    print(f"{key}: {value}")
'''