"""
Main script for calcualting the GIRF from the processed data. 
Beyond a few minor modifications this is a translation of the MATLAB code provided in https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html
See help_functions_GIRF for the calculation steps 
"""

import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import sys 
from help_functions_GIRF import calculate_output_gradient_optimized, display_girf_magnitude, read_multi_files, resamp_gradients
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.interpolate import CubicSpline

#Folder containing the .mat files 
dataPath = '/Users/jamesbacon/Library/CloudStorage/OneDrive-Nexus365/GIRF_Correction/250313_pe_girf/GIRF_PE_Modified'


dataSavePath = os.path.join(dataPath, 'Results6') #Creates Results folder

n=0 # Defines the number of cutoff points at the start to account for the ringing affect of the scanner (Using 0 is also generally fine)
f=20000 # Defines the number of points to include up until. The more points the greater the spectral resolution but also greater noise inclusion 

#Select the gradient axis as x,y,z
gradientAxis = 'z'.lower()
Linear = True # Set this to True to look at the Linear term of the GIRF and set it False ot look at the B0 term 
Save = False   # Sets if it is saved to file or not 

# Check the validities of user inputs
if gradientAxis not in ['x', 'y', 'z']:
    raise ValueError("Selected gradient axis must be 'X/x', 'Y/y', or 'Z/z'.")

# Setting data path and file names according to user selections
full_data_path = dataPath

#Change names based on filenames used 
# File paths based on the gradient axis
if gradientAxis == 'x':
    fn_Slice1_POS = 'Positive+xslice.mat'
    fn_Slice1_NEG = 'Negative+xslice.mat'
    fn_Slice2_POS = 'Positive-xslice.mat'
    fn_Slice2_NEG = 'Negative-xslice.mat'
    fn_Slice1_ref = 'Ref+xslice.mat'
    fn_Slice2_ref = 'Ref-xslice.mat'
elif gradientAxis == 'y':
    fn_Slice1_POS = 'Positive+yslice.mat'
    fn_Slice1_NEG = 'Negative+yslice.mat'
    fn_Slice2_POS = 'Positive-yslice.mat'
    fn_Slice2_NEG = 'Negative-yslice.mat'
    fn_Slice1_ref = 'Ref+yslice.mat'
    fn_Slice2_ref = 'Ref-yslice.mat'
else:  # Z-axis
    fn_Slice1_POS = 'Positive+zslice.mat'
    fn_Slice1_NEG = 'Negative+zslice.mat'
    fn_Slice2_POS = 'Positive-zslice.mat'
    fn_Slice2_NEG = 'Negative-zslice.mat'
    fn_Slice1_ref = 'Ref+zslice.mat'
    fn_Slice2_ref = 'Ref-zslice.mat'

# Load gradient waveforms
fn_gradient = 'InputGradients.mat'

# Full file paths
fn_slice1_pos = os.path.join(full_data_path, fn_Slice1_POS)
fn_slice1_neg = os.path.join(full_data_path, fn_Slice1_NEG)
fn_slice2_pos = os.path.join(full_data_path, fn_Slice2_POS)
fn_slice2_neg = os.path.join(full_data_path, fn_Slice2_NEG)
fn_slice1_ref = os.path.join(full_data_path, fn_Slice1_ref)
fn_slice2_ref = os.path.join(full_data_path, fn_Slice2_ref)
fn_gradient = os.path.join(full_data_path, fn_gradient)

#Change based on slice offset and thcikness used 
# Fixed parameters
params = {
    'gammabar': 42.576e3,  # in Hz/mT
    'slicePos': 0.017,     # distance of slices from isocenter in meters
    'gradRasterTime': 10,  # in microseconds
    'adcDwellTime': 5      # in microseconds
}


# Load necessary files
raw_sig_s1_pos, raw_sig_s1_neg, raw_sig_s2_pos, raw_sig_s2_neg, ref_s1, ref_s2 = read_multi_files(
    'kspace_all', fn_slice1_pos, fn_slice1_neg, fn_slice2_pos, fn_slice2_neg, fn_slice1_ref, fn_slice2_ref, N=n, F=f
)

#Load in input gradients 
grad_in_all = sio.loadmat(fn_gradient)['gradIn_all']


# Step 2: Processing gradient inputs

# Extract readout points, number of repetitions, and number of gradient blips from raw signals

#params['roPts'] = raw_sig_s1_pos.shape[0]  # Readout points
params['roPts'] = f-n
params['nRep'] = raw_sig_s1_pos.shape[1]   # Number of repetitions, this refers to number of PE (25=5*5)
params['nGradAmp'] = raw_sig_s1_pos.shape[2]  # Number of gradient blips (18)


# Resample gradients using the helper function `resamp_gradients`
gResamp, roTime = resamp_gradients(grad_in_all, params)

#Used later for calibration 
def shift_half_index_spline(array, shift):
    N = len(array)
    x = np.arange(N)
    spline = CubicSpline(x, array, bc_type='natural')
    x_shifted = x +shift   # Shift by half an index
    return spline(x_shifted)  # Evaluate shifted function

# The nominal starting time of the blips is 2000us after ADC starts
timeShift = 1990 - n*5 # in microseconds (1990 is used not 2000 due to ADC calibration )
if timeShift < 0:
    print('Error: timeShift is Negative')

gradInput = np.roll(gResamp, int(timeShift / params['adcDwellTime']), axis=0)


# Perform FFT, shift the zero frequency to the center, and replicate across nRep
gradInputFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(gradInput, axes=0), axis=0), axes=0)  # [nRO, nGradAmp]
#gradInputFT = np.tile(gradInputFT[:, :, np.newaxis], (1, 1, params['nRep']))  # Expand to [nRO, nGradAmp, nRep]
gradInputFT = np.tile(gradInputFT[:, :, np.newaxis], (1, 1, 1))


# Step 3: Calculate Output Gradients and GIRF

# Change dimension from [nRO, nRep, nGradAmp] to [nRO, nGradAmp, nRep]
rawSigS1_POS = np.transpose(raw_sig_s1_pos, (0, 2, 1))  # [nRO, nGradAmp, nRep]
rawSigS1_NEG = np.transpose(raw_sig_s1_neg, (0, 2, 1))
rawSigS2_POS = np.transpose(raw_sig_s2_pos, (0, 2, 1))
rawSigS2_NEG = np.transpose(raw_sig_s2_neg, (0, 2, 1))



# Calculate the output gradients using the optimized method
GIRFS = []

#Use just 12 for the central voxel. This is the middle of the phase encoding 
#To view mutiple GIRFS from the phase encoding jsut add the correct position 
for i in [12]:
    rawSigS1_POS1= rawSigS1_POS[:,:,i:i+1]
    rawSigS1_NEG1= rawSigS1_NEG[:,:,i:i+1]
    rawSigS2_POS1= rawSigS2_POS[:,:,i:i+1]
    rawSigS2_NEG1= rawSigS2_NEG[:,:,i:i+1]
    refS1 = ref_s1[:,i, :]
    refS2 = ref_s2[:,i, :]


    gradOutput, gradOutputFT = calculate_output_gradient_optimized(rawSigS1_POS1, rawSigS1_NEG1, rawSigS2_POS1, rawSigS2_NEG1, refS1, refS2, params, Linear = Linear )

    # Calculate the GIRF in the frequency domain
    # Sum up along the second axis (nGradAmp dimension)
    numerator = np.sum(gradOutputFT * np.conj(gradInputFT), axis=1)  # Element-wise multiply and sum
    denominator = np.sum(np.abs(gradInputFT) ** 2, axis=1)           # Sum of squared magnitudes
    GIRF_FT = numerator / denominator


    # Step 4: Plot GIRF in Frequency Domain

    # Calculate full frequency range of the GIRF spectrum
    freqRange = int(round(1 / (params['adcDwellTime'] / 1e6) / 1e3)  )# Full spectrum width, in kHz
    freqFull = np.linspace(-freqRange / 2 , freqRange / 2, params['roPts'])
    #Adjust the position due to small error using linspace
    freqFull = shift_half_index_spline(freqFull, -0.5)

    dispFreqRange = np.array([-30, 30])  # in unit of kHz

    display_girf_magnitude(GIRF_FT, freqFull, dispFreqRange, Linear = Linear )

    GIRFS.append(GIRF_FT[:, 0])

GIRFS= np.transpose(GIRFS, (1,0))

# Call displayGIRFMagnitude function to plot the GIRF magnitude spectrum

plt.show()


# Step 5: Save Results

# Check if the directory exists, if not, create it
if not os.path.exists(dataSavePath):
    os.makedirs(dataSavePath)

# Save results to the specified path
if Linear == True:
    file_name = f'GIRFOptimized_G{gradientAxis.upper()}Linear.mat'
    file_path = os.path.join(dataSavePath, file_name)
else: 
    file_name = f'GIRFOptimized_G{gradientAxis.upper()}B0.mat'
    file_path = os.path.join(dataSavePath, file_name)

if Save == True: 
    # Save the required variables: GIRF_FT, params, and roTime
    sio.savemat(file_path, {'GIRF_FT': GIRFS, 'params': params, 'roTime': roTime, 'freqFull': freqFull})

    # Optional: Debugging output
    print(f"Results saved to: {file_path}")