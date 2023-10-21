import pandas as pd


def downsample_dataset(original_dataset, interpolated_dataset):
    original_dataset = pd.read_csv(original_dataset)
    missing_values_list = []
    
    # Load the second dataset from which you want to extract values
    other_dataset = pd.read_csv(interpolated_dataset)
    
    for column in original_dataset.columns[:-3]:
        missing_positions = original_dataset.index[original_dataset[column] == 0.0].tolist()
        

        # Retrieve and store the values from the other dataset
        missing_values = [other_dataset.at[position, column] for position in missing_positions]
        

        missing_values_list.extend(missing_values)
        
    return missing_values_list





def downsample_dataset1(original_dataset, interpolated_dataset):
    original_dataset = pd.read_csv(original_dataset)
    missing_values_list = []
    
    # Extract the first column name
    first_column = original_dataset.columns[0]

    # Find missing positions in the first column
    missing_positions = original_dataset.index[original_dataset[first_column] == 0.0].tolist()

    # Load the second dataset from which you want to extract values
    other_dataset = pd.read_csv(interpolated_dataset)

    # Retrieve and store the values from the other dataset for the first column
    missing_values = [other_dataset.at[position, first_column] for position in missing_positions]

    missing_values_list.extend(missing_values)

    return missing_values_list




# zero_filled = "./original_datasets/zero_filled.csv"
# akima_dataset = "./interpolated_datasets/akima_dataset.csv"

# #call downsampling function
# missing_values_list = downsample_dataset(zero_filled, akima_dataset)