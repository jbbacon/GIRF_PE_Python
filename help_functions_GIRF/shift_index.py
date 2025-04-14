from scipy.interpolate import CubicSpline
import numpy as np

def shift_half_index_spline(array, shift):
    N = len(array)
    x = np.arange(N)
    spline = CubicSpline(x, array, bc_type='natural')
    x_shifted = x +shift   # Shift by half an index
    return spline(x_shifted)  # Evaluate shifted function