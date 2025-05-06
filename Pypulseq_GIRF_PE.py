"""
Pypulseq code to generate the pulse sequence for GIRF calcualtion

Inspired by https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html for the optimised GIRF calculation 
and https://onlinelibrary.wiley.com/doi/10.1002/mrm.27902 for the 5*5 phase encoding

Scan time is approximately 2 hours per direction for full usage

Terminal Command: pixi run gen-seq --direction x --output /path/to/output/folder
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='GIRF Sequence Generation.')
parser.add_argument('--direction', choices=['x', 'y', 'z'], default='z', required=True, help='Primary GIRF direction')
parser.add_argument('--output', type=Path, required=True, help='Output folder to save .seq file and additional parameter requirements')
parser.add_argument('--batch_size', type=int, default=3, help='Number of triangle pulses used between references')
parser.add_argument('--n', type=int, default=5, help='Number of phase encodes (Nx = Ny = n)')
parser.add_argument('--slice_thickness', type=float, default=1, help='Slice thickness in millimeters')
parser.add_argument('--fov', type=float, default=200e-3, help='Field of view in meters')
parser.add_argument('--slice_offset', type=float, default=17, help='Magnitude of slice offset in millimeters (positive and negative will be used)')
parser.add_argument('--num_triangles', type=int, choices=[6, 9, 18], default=18, help='Number of triangle gradients (6, 9, or 18)')
parser.add_argument('--no_save', action='store_false', dest='save', help='Flag to not save output files.')
parser.add_argument('--plot', action='store_true', help='Plot the sequence timing diagram if specified.')
parser.add_argument('--plot_range', type=float, nargs=2, metavar=('START', 'END'), default=(0, 200),help='Time range for plotting in seconds, used only if --plot is specified.')
args = parser.parse_args()


# Imports after argparse for speed of inital use
import pypulseq as pp
import numpy as np
import itertools
import csv
import os
import json

# Make output directory
args.output.mkdir(exist_ok=True, parents=True)

Nx = Ny = args.n
fov = args.fov
slice_thickness = args.slice_thickness/1000
slice_offset = args.slice_offset/1000
slice_offsets = [slice_offset, -slice_offset]
deltak = 1 / fov

# Set directions and FOV ordering
if args.direction == 'z':
    directions = ['z', 'x', 'y']
    fov_vector = [fov, fov, 2*slice_offset]
elif args.direction == 'y':
    directions = ['y', 'z', 'x']
    fov_vector = [fov, 2*slice_offset, fov]
else:
    directions = ['x', 'y', 'z']
    fov_vector = [2*slice_offset, fov, fov]

# Log for pulse ordering files
def write_log(pulse_log, log_file):
    fieldnames = ['batch_index', 'type', 'j', 'k', 'slice_offset', 'amplitude_sign', 'triangular_amplitude_mT/m']
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(pulse_log)

# Saving additional parameters needed for analysis 
def save_parameters(triangular_amplitudes, Nx, output_folder, save_flag):
    if save_flag:
        # Create a dictionary to store the triangular amplitudes and phase encode steps
        data = {
            'triangular_amplitudes': np.array(triangular_amplitudes).tolist(),
            'n': Nx,  # Store the number of phase encode steps (n)
            'slice_offset': slice_offset,
            'batch_size': args.batch_size
        }
        
        os.makedirs(output_folder, exist_ok=True)

        with open(output_folder / 'parameters.json', 'w') as fp:
            json.dump(data, fp)

        print(f"Parameters saved to {output_folder / 'parameters.json'}")
    else:
        print("Save flag is set to False. Skipping saving tof parameters.")

def ref(seq, system, slice_offsets, directions, log_file, batch_index, save_flag):
    pulse_log = []
    for j, k, slice_offset in itertools.product(range(Nx), range(Ny), slice_offsets):
        rf, gz, gzReph = pp.make_sinc_pulse(
            flip_angle=np.deg2rad(90),
            duration=4e-3,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=4,
            system=system,
            return_gz=True
        )
        gz.channel = directions[0]
        gzReph.channel = directions[0]
        rf.freq_offset = slice_offset * gz.amplitude

        gxPE = pp.make_trapezoid(channel=directions[1], area=(j - (Nx - 1)/2) * deltak,
                                 duration=pp.calc_duration(gzReph), system=system)
        gyPE = pp.make_trapezoid(channel=directions[2], area=(k - (Ny - 1)/2) * deltak,
                                 duration=pp.calc_duration(gzReph), system=system)

        adc = pp.make_adc(num_samples=80000, dwell=5e-6, system=system)
        gz_spoil = pp.make_trapezoid(channel=gz.channel, area=-gz.area / 2, system=system)

        seq.add_block(rf, gz)
        seq.add_block(gzReph, gxPE, gyPE)
        seq.add_block(adc)
        seq.add_block(gz_spoil)
        seq.add_block(pp.make_delay(3))

        pulse_log.append({
            'batch_index': batch_index,
            'type': 'ref',
            'j': j,
            'k': k,
            'slice_offset': slice_offset,
            'amplitude_sign': '',
            'triangular_amplitude_mT/m': ''
        })

    if save_flag:
        write_log(pulse_log, log_file)

def triangle(seq, system, triangular_amplitude_mT_per_m, slice_offsets, directions, log_file, batch_index, save_flag):
    triangular_amplitude_hz_per_m = triangular_amplitude_mT_per_m * system.gamma / 1000
    rise_time = triangular_amplitude_hz_per_m / system.max_slew
    pulse_log = []
    for j, k, amplitude_sign, offset in itertools.product(range(Nx), range(Ny), [1, -1], slice_offsets):
        rf, gz, gzReph = pp.make_sinc_pulse(
            flip_angle=np.deg2rad(90),
            duration=4e-3,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=4,
            system=system,
            return_gz=True
        )
        gz.channel = directions[0]
        gzReph.channel = directions[0]
        rf.freq_offset = offset * gz.amplitude

        gxPE = pp.make_trapezoid(channel=directions[1], area=(j - (Nx - 1)/2) * deltak,
                                 duration=pp.calc_duration(gzReph), system=system)
        gyPE = pp.make_trapezoid(channel=directions[2], area=(k - (Ny - 1)/2) * deltak,
                                 duration=pp.calc_duration(gzReph), system=system)

        adc = pp.make_adc(num_samples=80000, dwell=5e-6, system=system)
        grad_spoilz = pp.make_trapezoid(channel=directions[0], area=-gz.area / 2, system=system)
        grad_triangular = pp.make_trapezoid(
            channel=directions[0],
            amplitude=amplitude_sign * triangular_amplitude_hz_per_m,
            flat_time=0,
            rise_time=rise_time,
            fall_time=rise_time,
            delay=2e-3,
            system=system
        )
        seq.add_block(rf, gz)
        seq.add_block(gzReph, gxPE, gyPE)
        seq.add_block(adc, grad_triangular)
        seq.add_block(grad_spoilz)
        seq.add_block(pp.make_delay(3))

        pulse_log.append({
            'batch_index': batch_index,
            'type': 'triangle',
            'j': j,
            'k': k,
            'slice_offset': offset,
            'amplitude_sign': amplitude_sign,
            'triangular_amplitude_mT/m': triangular_amplitude_mT_per_m
        })

    if save_flag:
        write_log(pulse_log, log_file)

def create_triangular_wave(amplitude, step_size=1.8, length=4000):

    # Determine the number of rise and fall steps
    num_steps = round(amplitude / step_size)  # Number of steps to reach the maximum amplitude
    
    # Create the rising and falling part of the waveform
    rising_part = np.linspace(0, amplitude, num_steps + 1)
    falling_part = np.linspace(amplitude, 0, num_steps + 1)
    
    # Combine the rising and falling parts to form the full triangular wave
    full_wave = np.concatenate([rising_part, falling_part[1:]])  # Avoid duplicate peak value
    
    # Check if the waveform is shorter than the required length
    if len(full_wave) < length:
        # Zero pad to make it exactly the required length
        full_wave = np.pad(full_wave, (0, length - len(full_wave)), mode='constant', constant_values=0)
    else:
        # If it's longer, trim it to the desired length
        full_wave = full_wave[:length]
    
    return full_wave

def save_triangular_waves(triangular_amplitudes, output_folder, save_flag):
    if save_flag:
        # Create a list to store all the triangular waveforms
        all_triangles_matrix = np.asarray(
            [create_triangular_wave(amp) for amp in triangular_amplitudes]).T
        
        np.savez_compressed(output_folder / 'InputGradients.npz', gradIn_all = all_triangles_matrix)

        print(f"Triangular waves saved to {output_folder / 'InputGradients.npz'}")
    else:
        print("Save flag is set to False. Skipping saving the .npz file.")

# Validate batch size compatibility
if args.num_triangles % args.batch_size != 0:
    raise ValueError("Number of triangle gradients must be divisible by batch size.")

# Set amplitudes
full_amplitudes = [
    9, 10.8, 12.6, 14.4, 16.2, 18, 19.8, 21.6, 23.4,
    25.2, 27, 28.8, 30.6, 32.4, 34.2, 36, 37.8, 39.6
]

# Select subset of amplitudes based on user input
if args.num_triangles == 18:
    triangular_amplitudes = full_amplitudes
elif args.num_triangles == 9:
    triangular_amplitudes = full_amplitudes[::2]  # Every other one
elif args.num_triangles == 6:
    triangular_amplitudes = full_amplitudes[::3]  # Every third one
else:
    raise ValueError("num_triangles must be one of 6, 9, or 18")

# Save the triangular waveforms to a .npz file if --save is True
save_triangular_waves(triangular_amplitudes, args.output, args.save)

# Init sequence
seq = pp.Sequence()
system = pp.Opts(max_grad=40, grad_unit='mT/m', max_slew=180, slew_unit='T/m/s',
                 rf_ringdown_time=20e-6, rf_dead_time=100e-6,
                 grad_raster_time=10e-6, adc_dead_time=1e-5)

# File naming and paths
direction_letter = args.direction
os.makedirs(args.output, exist_ok=True)
log_file = os.path.join(args.output, f"pulse_order_log_{direction_letter}.csv")
if os.path.exists(log_file):
    os.remove(log_file)

# Build sequence
batch_index = 0
for i, amplitude in enumerate(triangular_amplitudes):
    if i % args.batch_size == 0:
        ref(seq, system, slice_offsets, directions, log_file, batch_index, args.save)
    triangle(seq, system, amplitude, slice_offsets, directions, log_file, batch_index, args.save)
    if i % args.batch_size == args.batch_size - 1:
        batch_index += 1

seq.set_definition(key='FOV', value=fov_vector)

if args.plot:
    seq.plot(time_range=(args.plot_range[0], args.plot_range[1]))

if args.save:
    seq_path = os.path.join(args.output, f"{direction_letter.upper()}Full.seq")
    seq.write(seq_path)
    print(f"Sequence written to {seq_path}")
else:
    print("Output not saved due to --save false.")

if args.save:
    save_parameters(triangular_amplitudes, Nx, args.output, args.save)