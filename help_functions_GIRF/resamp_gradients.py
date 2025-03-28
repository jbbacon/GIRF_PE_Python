import numpy as np
from scipy.interpolate import interp1d

def resamp_gradients(grad_in, params):
    """
    Resample input gradients to the ADC dwell time.

    Parameters:
    grad_in (ndarray): Input gradients with shape (nGradSamples, nGradIn)
    params (dict): A dictionary containing:
        - 'gradRasterTime': Time between gradient samples in microseconds
        - 'adcDwellTime': ADC dwell time in microseconds
        - 'roPts': Number of ADC samples (readout points)

    Returns:
    gResamp (ndarray): Resampled gradients with shape (roPts, nGradIn)
    roTime (ndarray): Time for each resampled gradient with shape (roPts,)
    """
    
    # Number of gradient samples
    nGradSamples = grad_in.shape[0]
    
    # Gradient time vector (in microseconds)
    gradTime = np.arange(nGradSamples) * params['gradRasterTime']
    
    # Readout time vector (in microseconds)
    roTime = np.arange(params['roPts']) * params['adcDwellTime']
    
    nGradIn = grad_in.shape[1]  # Number of gradient amplitudes
    
    # Initialize resampled gradients array
    gResamp = np.zeros((params['roPts'], nGradIn), dtype=np.float32)
    
    # Perform interpolation for each gradient amplitude
    for n in range(nGradIn):
        interpolator = interp1d(gradTime, grad_in[:, n], kind='linear', fill_value='extrapolate')
        gResamp[:, n] = interpolator(roTime)
    
    # Fix the possible interpolation error on the last point (if necessary)
    gResamp[-1, :] = 0
    roTime = roTime.reshape(-1,1)
    return gResamp, roTime
