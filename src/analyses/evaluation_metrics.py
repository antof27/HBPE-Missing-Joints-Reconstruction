import numpy as np
from scipy.spatial import distance


def calculate_rmse(gt_list, interpolated_list):
    """
    Calculates the Root Mean Squared Error (RMSE) between two lists of values.
    
    Parameters:
        gt_list (array-like): Ground truth values.
        interpolated_list (array-like): Interpolated values.
        
    Returns:
        float: RMSE value.
    """
    rmse = np.sqrt(np.mean((np.array(gt_list) - np.array(interpolated_list))**2))
    return rmse


def kl_divergence(p, q):
    # Convert distributions to numpy arrays
    p = np.array(p)
    q = np.array(q)
    
    # Ensure distributions have the same length
    if len(p) != len(q):
        raise ValueError("Distributions must have the same length.")
    
    # Avoid division by zero by adding a small value to each distribution
    p_nonzero = p + 1e-10
    q_nonzero = q + 1e-10
    
    # Compute KL divergence
    kl_div = np.sum(p_nonzero * np.log(p_nonzero / q_nonzero))
    
    return kl_div


def evaluation_metrics(gt_list, interpolated_list):
    """
    Calculates evaluation metrics between ground truth and interpolated values.
    
    Parameters:
        gt_list (array-like): Ground truth values.
        interpolated_list (array-like): Interpolated values.
        
    Returns:
        float: Euclidean distance between two lists.
        float: Root Mean Squared Error (RMSE) between two lists.
    """
    euclidean_dist = round(distance.euclidean(gt_list, interpolated_list), 3)
    rmse = calculate_rmse(gt_list, interpolated_list)
    #normalize the lists

    gt_list = np.array(gt_list) / np.sum(gt_list)
    interpolated_list = np.array(interpolated_list) / np.sum(interpolated_list)
    kl = kl_divergence(gt_list, interpolated_list)



    return euclidean_dist, rmse, kl
    



