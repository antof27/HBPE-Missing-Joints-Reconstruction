#create a function that read a dataset and flat all the columns appending all the columns values to a list
from downsampling import downsample_dataset
from evaluation_metrics import evaluation_metrics
import os 

# Get the current script's file path
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)
g_parent_directory = os.path.dirname(parent_directory)
data_directory = g_parent_directory + '/data/'


gt_dataset = data_directory + "selected_sequences_5.csv"
akima_dataset = data_directory + "akima_latest.csv"
zero_filled_dataset = data_directory + "zero_filled_latest.csv"
idw_dataset = data_directory + "idw_latest.csv"
linear_dataset = data_directory + "linear_latest.csv"
spline_dataset = data_directory + "spline_latest.csv"
pchip_dataset = data_directory + "pchip_latest.csv" 
nn_dataset = data_directory + "nearest_latest.csv"


akima_list = downsample_dataset(zero_filled_dataset, akima_dataset)
gt_list = downsample_dataset(zero_filled_dataset, gt_dataset)
idw_list = downsample_dataset(zero_filled_dataset, idw_dataset)
linear_list = downsample_dataset(zero_filled_dataset, linear_dataset)
spline_list = downsample_dataset(zero_filled_dataset, spline_dataset)
pchip_list = downsample_dataset(zero_filled_dataset, pchip_dataset)
nn_list = downsample_dataset(zero_filled_dataset, nn_dataset)


# Define your lists (gt_list, akima_list, idw_list, etc.) here

datasets = {
    "Akima": akima_list,
    "IDW": idw_list,
    "Linear": linear_list,
    "Spline": spline_list,
    "Pchip": pchip_list,
    "Nearest": nn_list
}

for dataset_name, dataset_list in datasets.items():
    euclid, rmse, kl = evaluation_metrics(gt_list, dataset_list)
    print("Dataset:", dataset_name)
    print("Euclidean Distance:", euclid)
    print("Root Mean Squared Percentage Error:", rmse)
    print("Kl Divergence: ", kl)
    print("\n")
    