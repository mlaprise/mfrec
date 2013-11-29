'''
.. container:: creation-info

    Created on August 11th, 2011
    
    @author Martin Laprise

Math library

'''

import numpy as np

def sigmoid(x, p1 = 1.0, scale_range = 4.0, y0 = 1.0, x0 = 0.0):
    y = scale_range / (1.0 + np.exp(-p1 * (x - x0))) + y0
    return y



