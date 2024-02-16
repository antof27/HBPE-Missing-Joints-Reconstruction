from scipy.interpolate import PchipInterpolator
import numpy as np

def pchip_interpolation(list1):

    missing_indices = np.where(np.array(list1) == 0)[0]
    list2 = np.delete(np.array(list1), missing_indices)

    x = np.arange(len(list1))
    y = np.delete(x, missing_indices)

    pchip_interpolation = PchipInterpolator(y, list2)

    for i in missing_indices:
        list1[i] = max(min(pchip_interpolation(i),1),0)

    
    

    return list1



list1 = [0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99]

list1 = pchip_interpolation(list1)

print(list1)
