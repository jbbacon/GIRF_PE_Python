import numpy as np
import matplotlib.pyplot as plt


def display_girf_magnitude(GIRF_FT, fullFreqRange, dispFreqRange=None, label=None):
    """
    Plots the magntidue of the GIRF Spherical harmoncis in the frequency domain 

    Args:
        GIRF_FT (np.array): Girf in the frequency domain, shape (nFreq, nSH)
        fullFreqRange (np.array): Full frequency range
        dispFreqRange (np.array, optional): Display frequency range. Defaults to None.
        label (str, optional): Label for the plot. Defaults to None.
    """
    # Check arguments and set display frequency range
    if dispFreqRange is None:
        dispFreqRange = [min(fullFreqRange), max(fullFreqRange)]
    

    GIRF_FT_mean = np.mean(GIRF_FT, axis=1)
    GIRF_FT_mean_abs = np.abs(GIRF_FT_mean)

    # Plot magnitude (mean and standard deviation)
    plt.plot(fullFreqRange, GIRF_FT_mean_abs, linewidth=1, label = label)
    plt.xlim(dispFreqRange)
    plt.ylim([0, 1.1])
    plt.xlabel('Frequency [Hz]', fontsize=10)
    plt.ylabel('Magnitude of GIRF [AU]', fontsize=10)
    plt.title('Magnitude of GIRF in Frequency Domain', fontsize=10)

