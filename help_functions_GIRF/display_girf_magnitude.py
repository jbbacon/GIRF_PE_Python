import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def display_girf_magnitude(GIRF_FT, fullFreqRange, dispFreqRange=None):
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

    
    # Fitting phase part for gradient temporal delay
    fitRange = [-9, 9]  # in kHz
    freqFitIndexStart = np.argmin(np.abs(fullFreqRange - fitRange[0]))
    freqFitIndexEnd = np.argmin(np.abs(fullFreqRange - fitRange[1]))
    freqFitIndex = np.arange(freqFitIndexStart, freqFitIndexEnd + 1)
    fitX = fullFreqRange[freqFitIndex]
    fitY = GIRF_FT_mean_phase[freqFitIndex]
    
    p1 = np.polyfit(fitX, fitY, 1)
    temporalDelay1 = p1[0] / (2 * np.pi) * 1e3  # in unit of microseconds
    #print(f'Gradient Delay is {temporalDelay1:.2f} us')

    """
    # Display phase (mean only)
    ax2 = fig.add_subplot(212)
    ax2.plot(fullFreqRange, GIRF_FT_mean_phase, 'b', linewidth=1)
    ax2.set_xlim(dispFreqRange)
    ax2.set_ylim([-4, 4])
    ax2.set_xlabel('Frequency [kHz]', fontsize=10)
    ax2.set_ylabel('Phase of GIRF [rad]', fontsize=10)
    ax2.set_title(f'Phase of GIRF in Frequency Domain, delay is {temporalDelay1:.2f} us', fontsize=10)
    """

   # plt.show()
