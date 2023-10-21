#create a function that read a dataset and flat all the columns appending all the columns values to a list
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from downsampling import downsample_dataset, downsample_dataset1
from evaluation_and_plotting.evaluation_metrics import evaluation_metrics
from evaluation_and_plotting.ecdf import ecdf, plotting_ecdf
from scipy.stats import gaussian_kde

gt_dataset = "./original_datasets/selected_sequences_5.csv"
akima_dataset = "./interpolated_datasets/nn_dataset.csv"
zero_filled_dataset = "./original_datasets/zero_filled.csv"
idw_dataset = "./interpolated_datasets/idw_dataset.csv"
linear_dataset = "./interpolated_datasets/linear_dataset.csv"
spline_dataset = "./interpolated_datasets/spline_dataset.csv"
pchip_dataset = "./interpolated_datasets/pchip_dataset.csv"
nn_dataset = "./interpolated_datasets/nn_dataset.csv"


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
    euclid, cosine, pearson, rmse = evaluation_metrics(gt_list, dataset_list)
    print(f"{dataset_name} :")
    print("Euclidean Distance:", euclid)
    print("Cosine Similarity:", cosine)
    print("Pearson Correlation Coefficient:", pearson)
    print("Root Mean Squared Error:", rmse)
    print("\n")


frames = list(range(len(gt_list)))  # Generate a list of frame indices

# Create NumPy arrays
array_l1 = np.array(gt_list)
array_l2 = np.array(akima_list)
array_l3 = np.array(idw_list)
array_l4 = np.array(linear_list)
array_l5 = np.array(spline_list)
array_l6 = np.array(pchip_list)
array_l7 = np.array(nn_list)

#create an array with all the lists
arrays = [array_l1, array_l2, array_l3, array_l4, array_l5, array_l6, array_l7]

#create a list with all the labels
labels = ['Ground Truth', 'Akima', 'Idw', 'Linear', 'Spline', 'Pchip', 'Nearest']

plotting_ecdf(arrays, labels)


#calculate the kde with epanechnikov kernel for all the lists


kde1 = gaussian_kde(array_l1, bw_method='silverman')
kde2 = gaussian_kde(array_l2, bw_method='silverman')
kde3 = gaussian_kde(array_l3, bw_method='silverman')
kde4 = gaussian_kde(array_l4, bw_method='silverman')
kde5 = gaussian_kde(array_l5, bw_method='silverman')
kde6 = gaussian_kde(array_l6, bw_method='silverman')
kde7 = gaussian_kde(array_l7, bw_method='silverman')

#these are the values over which your kernel will be evaluated
dist_space = np.linspace( min(array_l1), max(array_l1), 100 )

#plot the results
plt.plot( dist_space, kde1(dist_space), '-', label='Ground Truth')
plt.plot( dist_space, kde2(dist_space), '-', label='Akima')
plt.plot( dist_space, kde3(dist_space), '-', label='Idw')
plt.plot( dist_space, kde4(dist_space), '-.', label='Linear')
plt.plot( dist_space, kde5(dist_space), '-', label='Spline')
plt.plot( dist_space, kde6(dist_space), '-', label='Pchip')
plt.plot( dist_space, kde7(dist_space), '-', label='Nearest')

# Add labels and a legend
plt.xlabel('X Values')
plt.ylabel('Density')

plt.legend()

# Show the plot
plt.show()










'''



start = 10000
end = 10200

# plot only a portion of the data
frames = frames[start:end]
array_l1 = array_l1[start:end]
array_l2 = array_l2[start:end]
array_l3 = array_l3[start:end]
array_l4 = array_l4[start:end]
array_l5 = array_l5[start:end]
array_l6 = array_l6[start:end]
array_l7 = array_l7[start:end]


#plt.plot(frames, array_l1, 'ro', label='Ground Truth')
#plt.plot(frames, array_l2, 'go', label='Akima')
#plt.plot(frames, array_l3, 'bo', label='Idw')
plt.plot(frames, array_l4, '-.', label='Linear')
plt.plot(frames, array_l5, '--', label='Spline')
#plt.plot(frames, array_l6, 'yo', label='Pchip')
#plt.plot(frames, array_l7, 'ko', label='Nearest')




# Add labels and a legend
plt.xlabel('Frames')
plt.ylabel('X Values')
plt.legend()

# Show the plot
plt.show()
'''