import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr


def evaluation_metrics(gt_list, interpolated_list):
    euclidean_dist = round(distance.euclidean(gt_list, interpolated_list), 3)
    cosine_sim = round(1 - distance.cosine(gt_list, interpolated_list), 5)
    pearson_corr, _ = pearsonr(gt_list, interpolated_list)
    rmse = round(np.sqrt(np.mean(np.square(np.subtract(gt_list, interpolated_list)))),6)
    return euclidean_dist, cosine_sim, pearson_corr, rmse