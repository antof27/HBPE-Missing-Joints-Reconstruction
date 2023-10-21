import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def spline_interpolation(list1):

    missing_indices = np.where(np.array(list1) == 0)[0]
    list2 = np.delete(np.array(list1), missing_indices)

    x = np.arange(len(list1))
    y = np.delete(x, missing_indices)

    spl = CubicSpline(y, list2, bc_type=((1, 0.0), (1, 0.0)), extrapolate=True)

    for i in missing_indices:
        list1[i] = max(min(spl(i),1),0)
        #convert list[i] into a float va
        #interpolated_value = max(min(list[i], 1),0)
        #list1[i] = round(interpolated_value, 6)
    

    return list1

    
    
'''


list1 = [0.54, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.58]

list1 = spline_interpolation(list1)
print(list1)


#plot 
x = np.arange(len(list1))


plt.legend(loc='lower left')
plt.show()
'''
