"""
Main script for calcualting the GIRF from the processed data. 
Terminal Command: pixi run get-girf --data_path /path/to/data --direction x --json_file /path/to/json/file
Original inspiration of this code was a translation of the MATLAB code provided in https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html
See help_functions_GIRF for the calculation steps 
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from help_functions_GIRF import display_girf_magnitude, read_multi_files_np, resamp_gradients, calculate_output_gradient_optimized_spherical
from scipy.interpolate import CubicSpline
from matplotlib.widgets import Button
import argparse
import sys
import json

parser = argparse.ArgumentParser(description='Compute and save GIRF from gradient data.')

parser.add_argument('--data_path', type=str, required=True, help='Path to folder containing .npz input files.')
parser.add_argument('--json_file', type=str, required=True, help="Path to the parameters JSON file")
parser.add_argument('--direction', type=str, choices=['x', 'y', 'z'], required=True, help='Gradient axis (x, y, or z).')
parser.add_argument('--save', dest='save', action='store_true', help='Save results to Results Folder (default: True).')
parser.add_argument('--no_save', dest='save', action='store_false', help='Disable saving results.')
parser.set_defaults(save=True)
parser.add_argument('--plot', dest='plot', action='store_true', help='Display plot of GIRF (default: True).')
parser.add_argument('--no_plot', dest='plot', action='store_false', help='Disable plot display.')
parser.set_defaults(plot=True)
parser.add_argument('--order', type=int, default=1, help='Order of the Spherical Harmonics')
parser.add_argument('--save_path', type=str, default=None, help='Optional path to save output files. It will default to a results folder inside the input folder.')
parser.add_argument('--n', type=int, default=6, help='Number of cutoff points to account for ringing effects.')
parser.add_argument('--f', type=int, default=29999, help='Number of data points to include (resolution vs noise tradeoff).')
parser.add_argument('--min_thresh', type=float, default=0.001, help='Minimum threshold for voxel selection. (Increase to exclude more).')
parser.add_argument('--max_thresh', type=float, default=0.04, help='Maximum threshold for voxel selection. (Increase to exclude more).')

args = parser.parse_args()

dataPath = args.data_path
gradientAxis = args.direction.lower()
Save = args.save
Plotting = args.plot
n = args.n
f = args.f
order = args.order
min_threshold = args.min_thresh
max_threshold = args.max_thresh
json_file = args.json_file

def load_parameters(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if all(k in data for k in ['triangular_amplitudes', 'n', 'slice_offsets', 'fov']):
        slice_offsets = data['slice_offsets']
        fov = data['fov']
        return slice_offsets, fov
    else:
        raise KeyError("JSON missing required keys.")

slice_offsets, fov= load_parameters(json_file)

signs=[]
abs_mm=[]
for idx in range(4):
    slice_mm = slice_offsets[idx]
    sign = '+' if slice_mm > 0 else '-'
    abs1 = abs(slice_mm)
    abs_mm.append(abs1)
    signs.append(sign)

abs_mm=np.asarray(abs_mm)


if order not in (1, 2, 3):
    print('order must be 1, 2, or 3')
    sys.exit()

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

fn_Slice1_POS = f'Positive{signs[0]}{int(abs_mm[0]*1000)}{gradientAxis}slice.npz'
fn_Slice1_NEG = f'Negative{signs[0]}{int(abs_mm[0]*1000)}{gradientAxis}slice.npz'
fn_Slice2_POS = f'Positive{signs[1]}{int(abs_mm[1]*1000)}{gradientAxis}slice.npz'
fn_Slice2_NEG = f'Negative{signs[1]}{int(abs_mm[1]*1000)}{gradientAxis}slice.npz'
fn_Slice1_ref = f'Ref{signs[0]}{gradientAxis}_{int(abs_mm[0]*1000)}_slice.npz'
fn_Slice2_ref = f'Ref{signs[1]}{gradientAxis}_{int(abs_mm[1]*1000)}_slice.npz'
fn_Slice3_POS = f'Positive{signs[2]}{int(abs_mm[2]*1000)}{gradientAxis}slice.npz'
fn_Slice3_NEG = f'Negative{signs[2]}{int(abs_mm[2]*1000)}{gradientAxis}slice.npz'
fn_Slice4_POS = f'Positive{signs[3]}{int(abs_mm[3]*1000)}{gradientAxis}slice.npz'
fn_Slice4_NEG = f'Negative{signs[3]}{int(abs_mm[3]*1000)}{gradientAxis}slice.npz'
fn_Slice3_ref = f'Ref{signs[2]}{gradientAxis}_{int(abs_mm[2]*1000)}_slice.npz'
fn_Slice4_ref = f'Ref{signs[3]}{gradientAxis}_{int(abs_mm[3]*1000)}_slice.npz'


# Load gradient waveforms
fn_gradient = 'InputGradients.npz'

# Full file paths
fn_slice1_pos = os.path.join(full_data_path, fn_Slice1_POS)
fn_slice1_neg = os.path.join(full_data_path, fn_Slice1_NEG)
fn_slice2_pos = os.path.join(full_data_path, fn_Slice2_POS)
fn_slice2_neg = os.path.join(full_data_path, fn_Slice2_NEG)
fn_slice1_ref = os.path.join(full_data_path, fn_Slice1_ref)
fn_slice2_ref = os.path.join(full_data_path, fn_Slice2_ref)
fn_slice3_pos = os.path.join(full_data_path, fn_Slice3_POS)
fn_slice3_neg = os.path.join(full_data_path, fn_Slice3_NEG)
fn_slice4_pos = os.path.join(full_data_path, fn_Slice4_POS)
fn_slice4_neg = os.path.join(full_data_path, fn_Slice4_NEG)
fn_slice3_ref = os.path.join(full_data_path, fn_Slice3_ref)
fn_slice4_ref = os.path.join(full_data_path, fn_Slice4_ref)
fn_gradient = os.path.join(full_data_path, fn_gradient)


# Fixed parameters
params = {
    'gammabar': 42.576e3,  # in Hz/mT  
    'gradRasterTime': 10,  # in microseconds
    'adcDwellTime': 5      # in microseconds
}

# Load necessary files
raw_sig_s1_pos, raw_sig_s1_neg, raw_sig_s2_pos, raw_sig_s2_neg, ref_s1, ref_s2 = read_multi_files_np(
    'kspace_all', fn_slice1_pos, fn_slice1_neg, fn_slice2_pos, fn_slice2_neg, fn_slice1_ref, fn_slice2_ref, N=n, F=f
)


raw_sig_s3_pos, raw_sig_s3_neg, raw_sig_s4_pos, raw_sig_s4_neg, ref_s3, ref_s4 = read_multi_files_np(
    'kspace_all', fn_slice3_pos, fn_slice3_neg, fn_slice4_pos, fn_slice4_neg, fn_slice3_ref, fn_slice4_ref, N=n, F=f
)

refmax1 = np.mean(ref_s1, axis=2)
refmax2 = np.mean(ref_s2, axis=2)
refmax3 = np.mean(ref_s3, axis=2)
refmax4 = np.mean(ref_s4, axis=2)

#Load in input gradients 
with np.load(fn_gradient) as grad_data:
    grad_in_all = grad_data['gradIn_all']

with np.load(fn_slice1_pos) as slice_data:
    slice_offset1 = slice_data['slice_offset']
    dwell_time = slice_data['dwellTime']
    n2 = int(slice_data['n'])

with np.load(fn_slice2_pos) as slice_data:
    slice_offset2 = slice_data['slice_offset']

with np.load(fn_slice3_pos) as slice_data:
    slice_offset3 = slice_data['slice_offset']

with np.load(fn_slice4_pos) as slice_data:
    slice_offset4 = slice_data['slice_offset']

# Step 2: Processing gradient inputs
params['roPts'] = f-n
params['nRep'] = raw_sig_s1_pos.shape[1]   # Number of repetitions, this refers to number of PE (25=5*5)
params['nGradAmp'] = raw_sig_s1_pos.shape[2]  # Number of gradient blips (18)
params['slicePos1'] = slice_offset1
params['slicePos2'] = slice_offset2 
params['slicePos3'] = slice_offset3
params['slicePos4'] = slice_offset4     # distance of slices from isocenter in meters


def select_voxels(refmax, x, min_threshold = 0.01, max_threshold=0.01):
    max1 = []
    min1=[]
    for i in range(n2*n2):
        max1.append(max(np.abs(refmax[:, i])))
        min1.append(min(np.abs(refmax[:, i])))
    max1 = np.reshape(max1, (n2, n2))
    min1 = np.reshape(min1, (n2, n2))
    mask = (max1 > max_threshold/2) & (min1 > min_threshold/2)
    selected_indices = [tuple(idx) for idx in np.argwhere(mask)]

    padded_mask = np.pad(mask, 1, mode='constant', constant_values=False)

    confirmed = {'done': False}

    fig, axs = plt.subplots(n2, n2, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.1)
    fig.suptitle(f'Voxel Selection Slice {signs[x]}{int(abs_mm[x]*1000)}{gradientAxis}',fontsize=14)

    voxel_map = {}
    for i in range(n2):
        for j in range(n2):
            idx = i * n2 + j
            ax = axs[i, j]
            voxel_map[ax] = (i, j)
            ax.plot(np.abs(refmax[:, idx]), color='blue', linewidth=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(-0.01, 0.1)



            # Draw red outline if part of the mask
            pi, pj = i + 1, j + 1
            if mask[i, j]:
                if not padded_mask[pi - 1, pj]:
                    ax.plot([0, 1], [1, 1], transform=ax.transAxes, color='red', linewidth = 3)
                if not padded_mask[pi + 1, pj]:
                    ax.plot([0, 1], [0, 0], transform=ax.transAxes, color='red', linewidth = 3)
                if not padded_mask[pi, pj - 1]:
                    ax.plot([0, 0], [0, 1], transform=ax.transAxes, color='red', linewidth = 3)
                if not padded_mask[pi, pj + 1]:
                    ax.plot([1, 1], [0, 1], transform=ax.transAxes, color='red', linewidth = 3)
                
            if (i, j) in selected_indices:
                rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    color='green', linewidth=6, fill=False)
                ax.add_patch(rect)

    def on_click(event):
        for ax in voxel_map:
            if ax == event.inaxes:
                i, j = voxel_map[ax]
                if (i, j) in selected_indices:
                    # Unselect: remove from list and remove the green box
                    selected_indices.remove((i, j))
                    for patch in ax.patches[:]:
                        patch.remove()
                else:
                    # Select: add to list and draw green box
                    selected_indices.append((i, j))
                    rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                        color='green', linewidth=6, fill=False)
                    ax.add_patch(rect)
                fig.canvas.draw()
                break

    def on_confirm(event):
        confirmed['done'] = True
        plt.close(fig)

    # Confirm button
    button_ax = plt.axes([0.4, 0.01, 0.2, 0.05])
    button = Button(button_ax, 'Confirm')
    button.on_clicked(on_confirm)

    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    # Wait for user to confirm
    while not confirmed['done']:
        plt.pause(0.1)

    flat_indices = [i * n2 + j for (i, j) in selected_indices]
    return flat_indices

flat1 = select_voxels(refmax =refmax1, x=0 ,min_threshold=min_threshold, max_threshold=max_threshold)
flat2 = select_voxels(refmax =refmax2, x=1 ,min_threshold=min_threshold, max_threshold=max_threshold)
flat3 = select_voxels(refmax =refmax3, x=2 ,min_threshold=min_threshold, max_threshold=max_threshold)
flat4 = select_voxels(refmax =refmax4, x=3 ,min_threshold=min_threshold, max_threshold=max_threshold)


def shift_half_index_spline(array, shift):
    N = len(array)
    x = np.arange(N)
    spline = CubicSpline(x, array, bc_type='natural')
    x_shifted = x +shift   # Shift by half an index
    return spline(x_shifted)  # Evaluate shifted function

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


rawSigS1_POS = np.transpose(raw_sig_s1_pos, (0, 2, 1))  # [nRO, nGradAmp, nRep]
rawSigS1_NEG = np.transpose(raw_sig_s1_neg, (0, 2, 1))
rawSigS2_POS = np.transpose(raw_sig_s2_pos, (0, 2, 1))
rawSigS2_NEG = np.transpose(raw_sig_s2_neg, (0, 2, 1))
rawSigS3_POS = np.transpose(raw_sig_s3_pos, (0, 2, 1))  # [nRO, nGradAmp, nRep]
rawSigS3_NEG = np.transpose(raw_sig_s3_neg, (0, 2, 1))
rawSigS4_POS = np.transpose(raw_sig_s4_pos, (0, 2, 1))
rawSigS4_NEG = np.transpose(raw_sig_s4_neg, (0, 2, 1))

coeffs, coeffsFT = calculate_output_gradient_optimized_spherical(rawSigS1_POS, rawSigS1_NEG, rawSigS2_POS, rawSigS2_NEG, ref_s1, ref_s2, 
                                                                     rawSigS3_POS, rawSigS3_NEG, rawSigS4_POS, rawSigS4_NEG, ref_s3, ref_s4,
                                                                     params, gradientAxis=gradientAxis, order= order, index1= flat1, index2 = flat2, index3=flat3, index4=flat4, n2=n2, fov=fov )

girfs=[]

plt.figure(figsize=(6,4))
for i in range(len(coeffs[0,0,:])):
    numerator = np.sum(coeffsFT[:,:,i:i+1] * np.conj(gradInputFT[:,:, :]), axis = 1)
    denominator = np.sum(np.abs(gradInputFT[:,:, :])**2, axis=1)
    regularizer = 0

    GIRF_FT = numerator/(denominator+regularizer)

    freqRange = int(round(1 / (params['adcDwellTime'] / 1e6) / 1e3)  )# Full spectrum width, in kHz
    freqFull = np.linspace(-freqRange / 2 , freqRange / 2, params['roPts'])
    #Adjust the position due to small error using linspace

    if (f - n) % 2 == 0:
        freqFull = shift_half_index_spline(freqFull, -0.5)


    dispFreqRange = np.array([-30, 30])  # in unit of kHZ
    lbl = ['B0', 'x', 'y', 'z', 'xy', 'zy', '3z**2 - r**2', 'xz', 'x**2 - y**2', 'y(3x**2 - y**2)', 'xyz', 'y(5z**2 - r**2)', '5z**3 - 3zr**2', 'x(5z**2 - r**2)', 'z(x**2 - y**2)','x * (x**2 - 3y**2)']
    display_girf_magnitude(GIRF_FT, freqFull, dispFreqRange, label=lbl[i])

    girfs.append(GIRF_FT)

print(np.shape(girfs))

if Plotting ==True:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

if not os.path.exists(dataSavePath):
    os.makedirs(dataSavePath)


file_name = f'SphericalHarmonics_{gradientAxis.upper()}_{order}'
file_path = os.path.join(dataSavePath, file_name)
if Save == True: 
    # Save the required variables: GIRF_FT, params, and roTime

    np.savez(file_path, GIRF_FT=girfs, params=params, roTime=roTime, freqFull=freqFull)
    # Optional: Debugging output
    print(f"Results saved to: {file_path}")