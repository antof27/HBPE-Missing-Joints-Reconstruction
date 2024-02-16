import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import os 
# Get the current script's file path
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)
g_parent_directory = os.path.dirname(parent_directory)
data_directory = g_parent_directory + '/data/'

# Load the dataset
df_nn = pd.read_csv(data_directory + "nearest_latest.csv")
df_gt = pd.read_csv(data_directory + "selected_sequences_5.csv")
df_akima = pd.read_csv(data_directory + "akima_latest.csv")
df_idw = pd.read_csv(data_directory + "idw_latest.csv")
df_linear = pd.read_csv(data_directory + "linear_latest.csv")
df_spline = pd.read_csv(data_directory + "spline_latest.csv")
df_pchip = pd.read_csv(data_directory + "pchip_latest.csv")


row_nn = df_nn.iloc[172]
row_gt = df_gt.iloc[172]
row_akima = df_akima.iloc[172]
row_idw = df_idw.iloc[172]
row_linear = df_linear.iloc[172]
row_spline = df_spline.iloc[172]
row_pchip = df_pchip.iloc[172]

#all the columns are x,y and c values, use the x and y values to create a point into a plot 

x_nn = row_nn[0:][:74:3]
y_nn = row_nn[1:][:74:3]
y_nn = 1 - y_nn

x_akima = row_akima[0:][:74:3]
y_akima = row_akima[1:][:74:3]
y_akima = 1 - y_akima

x_idw = row_idw[0:][:74:3]
y_idw = row_idw[1:][:74:3]
y_idw = 1 - y_idw

x_linear = row_linear[0:][:74:3]
y_linear = row_linear[1:][:74:3]
y_linear = 1 - y_linear

x_spline = row_spline[0:][:74:3]
y_spline = row_spline[1:][:74:3]
y_spline = 1 - y_spline

x_pchip = row_pchip[0:][:74:3]
y_pchip = row_pchip[1:][:74:3]
y_pchip = 1 - y_pchip

x_gt = row_gt[0:][:74:3]
y_gt = row_gt[1:][:74:3]
y_gt = 1 - y_gt


colors = plt.cm.jet(np.linspace(0, 1, 25))
plt.scatter(x_gt, y_gt, c=colors, label='Ground Truth')
#plt.scatter(x_akima, y_akima, c=colors, marker='x', s=100, label='Akima')
#plt.scatter(x_nn, y_nn, c=colors, marker='x', s=100, label='Nearest Neighbour') 
#plt.scatter(x_idw, y_idw, c=colors, marker='x', s=100, label='IDW')
#plt.scatter(x_linear, y_linear, c=colors, marker='x', s=100, label='Linear')
#plt.scatter(x_spline, y_spline, c=colors, marker='x', s=100, label='Spline')
plt.scatter(x_pchip, y_pchip, c=colors, marker='x', s=100, label='Pchip')


plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()

plt.colorbar()

#save the plot as a png file 
plt.savefig('pchip_joints_plot.png', dpi=300)







