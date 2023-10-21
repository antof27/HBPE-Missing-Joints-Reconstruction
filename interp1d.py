from scipy.interpolate import interp1d
import numpy as np

def interp1d_interpolation(list1):

    missing_indices = np.where(np.array(list1) == 0)[0]
    list2 = np.delete(np.array(list1), missing_indices)

    x = np.arange(len(list1))
    y = np.delete(x, missing_indices)

    interp_1d = interp1d(y, list2, kind='nearest', bounds_error=False, fill_value=(0,1))

    for i in missing_indices:
        list1[i] = max(min(interp_1d(i),1),0)

    
    

    return list1

'''
#kind can be linear, nearest

#list1 = [0.54, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.58]
list1 = [0.54, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.58]

list1 = interp1d_interpolation(list1)

print(list1)
'''