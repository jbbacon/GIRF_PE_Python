import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def display_girf_magnitude(GIRF_FT, fullFreqRange, dispFreqRange=None, label=None):
    # Check arguments and set display frequency range
    if dispFreqRange is None:
        dispFreqRange = [min(fullFreqRange), max(fullFreqRange)]
    

    GIRF_FT_mean = np.mean(GIRF_FT, axis=1)
    GIRF_FT_std = np.std(GIRF_FT, axis=1)
    GIRF_FT_mean_abs = np.abs(GIRF_FT_mean)
    #GIRF_FT_mean_phase = np.angle(GIRF_FT_mean)
    GIRF_FT_std_abs = np.abs(GIRF_FT_std)
    

    # Plot magnitude (mean and standard deviation)

    errorBarColor = [255 / 255, 99 / 255, 71 / 255]  # Tomato Red Color
    plt.fill_between(fullFreqRange, 
                    GIRF_FT_mean_abs - GIRF_FT_std_abs, 
                    GIRF_FT_mean_abs + GIRF_FT_std_abs, 
                    color=errorBarColor, alpha=0.8)
    plt.plot(fullFreqRange, GIRF_FT_mean_abs, linewidth=1, label = label)
    plt.xlim(dispFreqRange)
    plt.ylim([0, 1.1])
    plt.xlabel('Frequency [kHz]', fontsize=10)
    plt.ylabel('Magnitude of GIRF [AU]', fontsize=10)
    plt.title('Magnitude of GIRF in Frequency Domain', fontsize=10)

    # Add zoomed-in inset
    #axins = zoomed_inset_axes(ax1, zoom=6, loc=8)  # Zoom factor 5
    #oom_range = [-4, 4]  # Adjust to the desired frequency range
    #axins.fill_between(fullFreqRange, 
                    #GIRF_FT_mean_abs - GIRF_FT_std_abs, 
                    #GIRF_FT_mean_abs + GIRF_FT_std_abs, 
                    #color=errorBarColor, alpha=0.8)
    #axins.plot(fullFreqRange, GIRF_FT_mean_abs, 'b', linewidth=1)
    #axins.set_xlim(zoom_range)  # Zoomed-in frequency range
    #axins.set_ylim(0.95, 1.005)  # Adjust to zoomed-in Y-axis range

    # Remove inset ticks
    #axins.set_xticks([])
    #axins.set_yticks([])

    # Connect inset to main plot
    #mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

   # plt.show()
