import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import os 

# Get the current script's file path
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)
g_parent_directory = os.path.dirname(parent_directory)
data_directory = g_parent_directory + '/data/'





def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y

def plotting_ecdf(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7):
    plt.plot(x1, y1, '-', label='Akima')
    plt.plot(x2, y2, '-', label='IDW')
    plt.plot(x3, y3, '-', label='Linear')
    plt.plot(x4, y4, '-.', label='Nearest')
    plt.plot(x5, y5, '-', label='Pchip')
    plt.plot(x6, y6, '-', label='Spline')
    plt.plot(x7, y7, '-', label='Ground Truth')

    # Add labels and a legend
    plt.xlabel('X Values')
    plt.ylabel('ECDF')
    plt.xlim(0.000060, 0.00014)
    #plt.ylim(0.07, 0.3)
    plt.legend()

    # save the plot at high resolution
    plt.savefig('ecdf1.png', dpi=300)


df_gt = pd.read_csv(data_directory +'selected_sequences_5.csv')

df_akima = pd.read_csv(data_directory + 'akima_latest.csv')
df_nearest = pd.read_csv(data_directory + 'nearest_latest.csv')
df_idw = pd.read_csv(data_directory + 'idw_latest.csv')
df_linear = pd.read_csv(data_directory + 'linear_latest.csv')
df_spline = pd.read_csv(data_directory + 'spline_latest.csv')
df_pchip = pd.read_csv(data_directory + 'pchip_latest.csv')


#Extract the joints coordinate 
x_gt_values = []
y_gt_values = []

x_akima_values = []
y_akima_values = []

x_idw_values = []
y_idw_values = []

x_linear_values = []
y_linear_values = []

x_spline_values = []
y_spline_values = []

x_pchip_values = []
y_pchip_values = []

x_nearest_values = []
y_nearest_values = []

joint = 9

x_gt_values = df_gt.iloc[1:, joint]
y_gt_values = df_gt.iloc[1:, joint+1]
y_gt_values = 1 - y_gt_values



x_akima_values = df_akima.iloc[1:, joint]
y_akima_values = df_akima.iloc[1:, joint+1]
y_akima_values = 1 - y_akima_values

x_idw_values = df_idw.iloc[1:, joint]
y_idw_values = df_idw.iloc[1:, joint+1]
y_idw_values = 1 - y_idw_values

x_linear_values = df_linear.iloc[1:, joint]
y_linear_values = df_linear.iloc[1:, joint+1]
y_linear_values = 1 - y_linear_values

x_spline_values = df_spline.iloc[1:, joint]
y_spline_values = df_spline.iloc[1:, joint+1]
y_spline_values = 1 - y_spline_values

x_pchip_values = df_pchip.iloc[1:, joint]
y_pchip_values = df_pchip.iloc[1:, joint+1]
y_pchip_values = 1 - y_pchip_values

x_nearest_values = df_nearest.iloc[1:, joint]
y_nearest_values = df_nearest.iloc[1:, joint+1]
y_nearest_values = 1 - y_nearest_values


#plot all the distributions in the same plot
frames = np.linspace(0, len(x_gt_values), len(x_gt_values))

plt.plot(frames, y_gt_values, '--', label='Ground Truth')
plt.plot(frames, y_akima_values, '-', label='Akima')
plt.plot(frames, y_idw_values, '-', label='Idw')
plt.plot(frames, y_linear_values, '-.', label='Linear')
plt.plot(frames, y_spline_values, '-', label='Spline')
plt.plot(frames, y_pchip_values, '-', label='Pchip')
plt.plot(frames, y_nearest_values, '-', label='Nearest')


# Add labels and a legend
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.xlim(3050,3200)
plt.ylim(0.56, 0.70)
plt.legend()

plt.savefig('scatter2.png', dpi=300)




#normalize the x and y values of all the interpolations
x_gt_values = np.array(x_gt_values) / np.sum(x_gt_values)
y_gt_values = np.array(y_gt_values) / np.sum(y_gt_values)

x_akima_values = np.array(x_akima_values) / np.sum(x_akima_values)
y_akima_values = np.array(y_akima_values) / np.sum(y_akima_values)


x_idw_values = np.array(x_idw_values) / np.sum(x_idw_values)
y_idw_values = np.array(y_idw_values) / np.sum(y_idw_values)

x_linear_values = np.array(x_linear_values) / np.sum(x_linear_values)
y_linear_values = np.array(y_linear_values) / np.sum(y_linear_values)

x_spline_values = np.array(x_spline_values) / np.sum(x_spline_values)
y_spline_values = np.array(y_spline_values) / np.sum(y_spline_values)

x_pchip_values = np.array(x_pchip_values) / np.sum(x_pchip_values)
y_pchip_values = np.array(y_pchip_values) / np.sum(y_pchip_values)

x_nearest_values = np.array(x_nearest_values) / np.sum(x_nearest_values)
y_nearest_values = np.array(y_nearest_values) / np.sum(y_nearest_values)

#print the shape of all the values
print(x_gt_values.shape)
print(y_gt_values.shape)
print(x_akima_values.shape)
print(y_akima_values.shape)
print(x_idw_values.shape)
print(y_idw_values.shape)
print(x_linear_values.shape)
print(y_linear_values.shape)
print(x_spline_values.shape)
print(y_spline_values.shape)
print(x_pchip_values.shape)
print(y_pchip_values.shape)
print(x_nearest_values.shape)
print(y_nearest_values.shape)



#calculate the ecdf for all the interpolations
x_gt, y_gt = ecdf(x_gt_values)
x_akima, y_akima = ecdf(x_akima_values)
x_idw, y_idw = ecdf(x_idw_values)
x_linear, y_linear = ecdf(x_linear_values)
x_spline, y_spline = ecdf(x_spline_values)
x_pchip, y_pchip = ecdf(x_pchip_values)
x_nearest, y_nearest = ecdf(x_nearest_values)


plotting_ecdf(x_akima, y_akima, x_idw, y_idw, x_linear, y_linear, x_nearest, y_nearest, x_pchip, y_pchip, x_spline, y_spline, x_gt, y_gt)


#plot kde for all the interpolations
kde1 = gaussian_kde(x_gt_values, bw_method='silverman')
kde2 = gaussian_kde(x_akima_values, bw_method='silverman')
kde3 = gaussian_kde(x_idw_values, bw_method='silverman')
kde4 = gaussian_kde(x_linear_values, bw_method='silverman')
kde5 = gaussian_kde(x_spline_values, bw_method='silverman')
kde6 = gaussian_kde(x_pchip_values, bw_method='silverman')
kde7 = gaussian_kde(x_nearest_values, bw_method='silverman')

#these are the values over which your kernel will be evaluated
dist_space = np.linspace( min(x_gt_values), max(x_gt_values), 100 )

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
plt.xlim(0.00008, 0.00009)
plt.ylim(0, 10000)

plt.legend()

plt.savefig('kde2.png', dpi=300)


