import numpy as np
import matplotlib.pyplot as plt

def ecdf(data):
    """Compute the ECDF of a one-dimensional array of data."""
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

def plotting_ecdf(arrays, labels):
    """
    Plot the ECDF for multiple arrays with specified labels.

    Args:
    arrays (list of arrays): List of data arrays.
    labels (list of str): List of labels for the arrays.

    Example:
    plotting_ecdf([array_l1, array_l2, array_l3, array_l4, array_l5, array_l6, array_l7],
                  ['Ground Truth', 'Akima', 'Idw', 'Linear', 'Spline', 'Pchip', 'Nearest'])
    """
    plt.figure(figsize=(10, 6))  # Define the figure size

    for i, array in enumerate(arrays):
        x, y = ecdf(array)
        plt.plot(x, y, label=labels[i])

    # Add labels and a legend
    plt.xlabel('X Values')
    plt.ylabel('ECDF')
    plt.legend()

    # Show the plot
    plt.show()
    
    '''
    plt.plot(x1, y1, '-', label='Ground Truth')
    plt.plot(x2, y2, '-', label='Akima')
    plt.plot(x3, y3, '-', label='Idw')
    plt.plot(x4, y4, '-.', label='Linear')
    plt.plot(x5, y5, '-', label='Spline')
    plt.plot(x6, y6, '-', label='Pchip')
    plt.plot(x7, y7, '-', label='Nearest')

    # Add labels and a legend
    plt.xlabel('X Values')
    plt.ylabel('ECDF')
    plt.legend()

    # Show the plot
    plt.show()
    '''