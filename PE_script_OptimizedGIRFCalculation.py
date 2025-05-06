"""
Main script for calcualting the GIRF from the processed data. 
Terminal Command: pixi run get-girf --data_path /path/to/data --direction x
Beyond a few minor modifications this is a translation of the MATLAB code provided in https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html
See help_functions_GIRF for the calculation steps 
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from help_functions_GIRF import calculate_output_gradient_optimized, display_girf_magnitude, read_multi_files, resamp_gradients, shift_half_index_spline
import argparse

parser = argparse.ArgumentParser(description='Compute and save GIRF from gradient data.')

parser.add_argument('--data_path', type=str, required=True, help='Path to folder containing .npz input files.')
parser.add_argument('--direction', type=str, choices=['x', 'y', 'z'], required=True, help='Gradient axis (x, y, or z).')
parser.add_argument('--save', dest='save', action='store_true', help='Save results to Results Folder (default: True).')
parser.add_argument('--no_save', dest='save', action='store_false', help='Disable saving results.')
parser.set_defaults(save=True)
parser.add_argument('--plot', dest='plot', action='store_true', help='Display plot of GIRF (default: True).')
parser.add_argument('--no_plot', dest='plot', action='store_false', help='Disable plot display.')
parser.set_defaults(plot=True)
parser.add_argument('--linear', dest='linear', action='store_true', help='Use linear term (default: True).')
parser.add_argument('--b0', dest='linear', action='store_false', help='Compute B0 term instead.')
parser.set_defaults(linear=True)
parser.add_argument('--save_path', type=str, default=None, help='Optional path to save output files. It will default to a results folder inside the input folder.')
parser.add_argument('--n', type=int, default=0, help='Number of cutoff points to account for ringing effects.')
parser.add_argument('--f', type=int, default=30000, help='Number of data points to include (resolution vs noise tradeoff).')

args = parser.parse_args()

dataPath = args.data_path
gradientAxis = args.direction.lower()
Linear = args.linear
Save = args.save
Plotting = args.plot
n = args.n
f = args.f

# Output path setup
if args.save_path:
    dataSavePath = args.save_path
else:
    dataSavePath = os.path.join(dataPath, f'Results{f-n}')


# Check the validities of user inputs
if gradientAxis not in ['x', 'y', 'z']:
    raise ValueError("Selected gradient axis must be 'X/x', 'Y/y', or 'Z/z'.")

# Setting data path and file names according to user selections
full_data_path = dataPath

# File paths based on the gradient axis
if gradientAxis == 'x':
    fn_Slice1_POS = 'Positive+xslice.npz'
    fn_Slice1_NEG = 'Negative+xslice.npz'
    fn_Slice2_POS = 'Positive-xslice.npz'
    fn_Slice2_NEG = 'Negative-xslice.npz'
    fn_Slice1_ref = 'Ref+xslice.npz'
    fn_Slice2_ref = 'Ref-xslice.npz'
elif gradientAxis == 'y':
    fn_Slice1_POS = 'Positive+yslice.npz'
    fn_Slice1_NEG = 'Negative+yslice.npz'
    fn_Slice2_POS = 'Positive-yslice.npz'
    fn_Slice2_NEG = 'Negative-yslice.npz'
    fn_Slice1_ref = 'Ref+yslice.npz'
    fn_Slice2_ref = 'Ref-yslice.npz'
else:  # Z-axis
    fn_Slice1_POS = 'Positive+zslice.npz'
    fn_Slice1_NEG = 'Negative+zslice.npz'
    fn_Slice2_POS = 'Positive-zslice.npz'
    fn_Slice2_NEG = 'Negative-zslice.npz'
    fn_Slice1_ref = 'Ref+zslice.npz'
    fn_Slice2_ref = 'Ref-zslice.npz'

# Load gradient waveforms
fn_gradient = 'InputGradients.npz'

# Full file paths
fn_slice1_pos = os.path.join(full_data_path, fn_Slice1_POS)
fn_slice1_neg = os.path.join(full_data_path, fn_Slice1_NEG)
fn_slice2_pos = os.path.join(full_data_path, fn_Slice2_POS)
fn_slice2_neg = os.path.join(full_data_path, fn_Slice2_NEG)
fn_slice1_ref = os.path.join(full_data_path, fn_Slice1_ref)
fn_slice2_ref = os.path.join(full_data_path, fn_Slice2_ref)
fn_gradient = os.path.join(full_data_path, fn_gradient)


# Fixed parameters
params = {
    'gammabar': 42.576e3,  # in Hz/mT
    'gradRasterTime': 10,  # in microseconds
    'adcDwellTime': 5      # in microseconds
}

# Load necessary files
raw_sig_s1_pos, raw_sig_s1_neg, raw_sig_s2_pos, raw_sig_s2_neg, ref_s1, ref_s2 = read_multi_files(
    'kspace_all', fn_slice1_pos, fn_slice1_neg, fn_slice2_pos, fn_slice2_neg, fn_slice1_ref, fn_slice2_ref, N=n, F=f
)

#Load in input gradients 
with np.load(fn_gradient) as grad_data:
    grad_in_all = grad_data['gradIn_all']

with np.load(fn_slice1_pos) as slice_data:
    slice_offset = slice_data['slice_offset']

with np.load(fn_slice1_pos) as batch_data:
    batch_size = batch_data['batch_size']

# Step 2: Processing gradient inputs
params['roPts'] = f-n
params['nRep'] = raw_sig_s1_pos.shape[1]   # Number of PE, this refers to number of PE (25=5*5)
params['nGradAmp'] = raw_sig_s1_pos.shape[2]  # Number of gradient blips (18)
params['slicePos'] = slice_offset     # distance of slices from isocenter in meters
params['batch_size'] = batch_size

# Resample gradients using the helper function `resamp_gradients`
gResamp, roTime = resamp_gradients(grad_in_all, params)

# The nominal starting time of the blips is 2000us after ADC starts
timeShift = 1990 - n*params['adcDwellTime'] # in microseconds (1990 is used not 2000 due to ADC calibration )
if timeShift < 0:
    raise ValueError("Error: timeShift is negative. Check value of n.")

gradInput = np.roll(gResamp, int(timeShift / params['adcDwellTime']), axis=0)


# Perform FFT, shift the zero frequency to the center, and replicate across nPE
gradInputFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(gradInput, axes=0), axis=0), axes=0)  # [nRO, nGradAmp]
gradInputFT = np.tile(gradInputFT[:, :, np.newaxis], (1, 1, 1))


# Step 3: Calculate Output Gradients and GIRF
# Change dimension from [nRO, nPE, nGradAmp] to [nRO, nGradAmp, nPE]
rawSigS1_POS = np.transpose(raw_sig_s1_pos, (0, 2, 1))  # [nRO, nGradAmp, nPE]
rawSigS1_NEG = np.transpose(raw_sig_s1_neg, (0, 2, 1))
rawSigS2_POS = np.transpose(raw_sig_s2_pos, (0, 2, 1))
rawSigS2_NEG = np.transpose(raw_sig_s2_neg, (0, 2, 1))

# Obtains the middle index corresponing to the centre of the slice
pre_index = (len(rawSigS1_NEG[0,0,:]))
index = int((pre_index-1)/2)

rawSigS1_POS1= rawSigS1_POS[:,:,index:index+1]
rawSigS1_NEG1= rawSigS1_NEG[:,:,index:index+1]
rawSigS2_POS1= rawSigS2_POS[:,:,index:index+1]
rawSigS2_NEG1= rawSigS2_NEG[:,:,index:index+1]
refS1 = ref_s1[:,index, :]
refS2 = ref_s2[:,index, :]


gradOutput, gradOutputFT = calculate_output_gradient_optimized(rawSigS1_POS1, rawSigS1_NEG1, rawSigS2_POS1, rawSigS2_NEG1, refS1, refS2, params, Linear = Linear)

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
if (f - n) % 2 == 0:
    freqFull = shift_half_index_spline(freqFull, -0.5)

dispFreqRange = np.array([-30, 30])  # in unit of kHz

display_girf_magnitude(GIRF_FT, freqFull, dispFreqRange, Linear = Linear )

# Call displayGIRFMagnitude function to plot the GIRF magnitude spectrum

if Plotting ==True:
    plt.show()

# Step 5: Save Results



# Set the output file name based on gradient type
if Linear:
    file_name = f'GIRFOptimized_G{gradientAxis.upper()}Linear.npz'
else:
    file_name = f'GIRFOptimized_G{gradientAxis.upper()}B0.npz'



if Save:

    # Check if the directory exists, if not, create it
    if not os.path.exists(dataSavePath):
        os.makedirs(dataSavePath)
        
    file_path = os.path.join(dataSavePath, file_name)
    # Save the required variables: GIRF_FT, params, roTime, freqFull
    np.savez(file_path, GIRF_FT=GIRF_FT, params=params, roTime=roTime, freqFull=freqFull)

    # Optional: Debugging output
    print(f"Results saved to: {file_path}")