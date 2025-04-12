import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def display_girf_magnitude(GIRF_FT, fullFreqRange, dispFreqRange=None, Linear= True):
    # Check arguments and set display frequency range
    if dispFreqRange is None:
        dispFreqRange = [min(fullFreqRange), max(fullFreqRange)]
    

    GIRF_FT_mean = np.mean(GIRF_FT, axis=1)
    GIRF_FT_std = np.std(GIRF_FT, axis=1)
    GIRF_FT_mean_abs = np.abs(GIRF_FT_mean)
    GIRF_FT_mean_phase = np.angle(GIRF_FT_mean)
    GIRF_FT_std_abs = np.abs(GIRF_FT_std)
    

    # Plot magnitude (mean and standard deviation)

    errorBarColor = [255 / 255, 99 / 255, 71 / 255]  # Tomato Red Color
    plt.fill_between(fullFreqRange, 
                    GIRF_FT_mean_abs - GIRF_FT_std_abs, 
                    GIRF_FT_mean_abs + GIRF_FT_std_abs, 
                    color=errorBarColor, alpha=0.8)
    plt.plot(fullFreqRange, GIRF_FT_mean_abs, linewidth=1)
    plt.xlim(dispFreqRange)
    if Linear ==True:
        plt.ylim([0, 1.1])
    else:
        plt.ylim(0, 0.05)
    plt.xlabel('Frequency [kHz]', fontsize=10)
    plt.ylabel('Magnitude of GIRF [AU]', fontsize=10)
    plt.title('Magnitude of GIRF in Frequency Domain', fontsize=10)
    