from scipy.interpolate import Akima1DInterpolator
import numpy as np
import math

def replace_nan_with_zero(input_list):
    return [0.0 if math.isnan(x) else x for x in input_list]


def akima_interpolation(list1):
    
    missing_indices = np.where(np.array(list1) == 0)[0]
    list2 = np.delete(np.array(list1), missing_indices)

    x = np.arange(len(list1))
    y = np.delete(x, missing_indices)

    akima = Akima1DInterpolator(y, list2)

    for i in missing_indices:
        list1[i] = max(min(akima(i),1),0)

    list1 = replace_nan_with_zero(list1)
    return list1

'''
#list1 = [0.54, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.58]
list1 = [0.0, 0.0, 0.0, 0.54, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.58]

list1 = akima_interpolation(list1)
list1 = replace_nan_with_zero(list1)
print(list1)
'''