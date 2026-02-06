#Run as pixi run proc-data --mri_file /path/to/mri_data.dat --csv_file /path/to/pulse_order_log_x.csv --json_file /path/to/parameters.json --npz_file /path/to/InputGradients.npz --direction x --output_folder /path/to/output/folder

import twixtools
import numpy as np
import pandas as pd
import json 
from fsl_mrs.utils.preproc.combine import svd_reduce, weightedCombination
import os
from pathlib import Path
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Process MRI raw data.")
    parser.add_argument("--mri_file", type=str, required=True, help="Path to the Siemens .dat file")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the pulse order log CSV file")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the parameters JSON file")
    parser.add_argument('--npz_file', type=str, required=True, help="Path to the Input Gradient .npz file")
    parser.add_argument("--output_folder", type=Path, required=True, help="Path to output folder")
    parser.add_argument('--direction', type=str, required=True, choices=['x', 'y', 'z'], help='Direction for slice labeling')
    parser.add_argument('--coils', type=int, default=32, help='Number of receiver coils (default: 32)')
    parser.add_argument('--hann_filter', action='store_true', help='Apply a radial Hann filter to data before 2D FT')
    parser.add_argument('--csi_filter', action='store_true', help='Apply CSI filter (cosine-weighted 2D window) before 2D FT')
    parser.add_argument("--n2", type=int, default= None, help="Final matrix size after zero padding")
    parser.add_argument("--dtype", type=str, choices=["complex128", "complex64"], default="complex128",help="Data type for processing (default: complex128), reduce for large file sizes")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dtype == "complex128":
        dtype = np.complex128
    elif args.dtype == "complex64":
        dtype = np.complex64
    mri_file = args.mri_file
    csv_file = args.csv_file
    json_file = args.json_file
    direction = args.direction
    coils = args.coils
    output_folder = args.output_folder

    if args.hann_filter and args.csi_filter:
        raise ValueError("Choose only one of --hann_filter or --csi_filter, not both.")

    # now carry on with your existing pipeline
    x = twixtools.map_twix(mri_file)
    img_obj = x[-1]['image']
    img_obj.flags['squeeze_singletons'] = True
    img_obj.flags['squeeze_ave_dims'] = False
 

    def analyze_batches_and_types(csv_filename):
        pulse_sequence_info = pd.read_csv(csv_filename)

        batch_indices = pulse_sequence_info['batch_index']

        batch_size = len(batch_indices) // len(batch_indices.unique())
        num_batches = len(batch_indices.unique())
        type_counts = pulse_sequence_info.groupby('batch_index')['type'].value_counts().unstack(fill_value=0)
        num_ref = type_counts.iloc[0].get('ref', 0)
        num_tri = type_counts.iloc[0].get('triangle', 0)
        return batch_size, num_batches, num_ref, num_tri

    batch_size, num_batches, num_ref, num_tri= analyze_batches_and_types(csv_file)

    def load_parameters(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        if all(k in data for k in ['triangular_amplitudes', 'n', 'slice_offsets', 'fov']):
            triangular_amplitudes = np.array(data['triangular_amplitudes'])
            n = data['n']
            slice_offsets = data['slice_offsets']
            return triangular_amplitudes, n, slice_offsets
        else:
            raise KeyError("JSON missing required keys.")


    triangular_amplitudes, n1, slice_offsets= load_parameters(json_file)

    # If n2 not supplied, default to n1
    n2 = args.n2 if args.n2 is not None else n1

    if n2 < n1 or (n2 - n1) % 2 != 0:
        raise ValueError(f"n2 must be >= n1 and differ by an even number. Got n1={n1}, n2={n2}")
    elif n1==n2:
        print('No Zero Padding')
    else:
        print(f'Zero padding from n={n1} to n={n2}')

    def make_radial_hann(n):
        assert n % 2 == 1, "Hann filter size n must be odd"
        radius = (n - 1) / 2
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        r = np.sqrt(x**2 + y**2)
        r_max = np.sqrt(2) * radius
        hann_radial = np.where(
            r <= r_max,
            0.5 * (1 + np.cos(np.pi * r / r_max)),
            0
        )
        return hann_radial

    def make_csi_filter(n, csi_averages=1):
        csi_size = np.array([n, n, 1])
        kmin = np.ceil(-csi_size / 2).astype(int)
        kmax = np.ceil(csi_size / 2 - 1).astype(int)

        x = np.arange(kmin[0], kmax[0] + 1)
        y = np.arange(kmin[1], kmax[1] + 1)
        z = np.arange(kmin[2], kmax[2] + 1)
        xvals, yvals, zvals = np.meshgrid(x, y, z, indexing='ij')

        dist = np.zeros_like(xvals, dtype=np.float64)
        if kmax[0] != 0:
            dist += (xvals / kmax[0])**2
        if kmax[1] != 0:
            dist += (yvals / kmax[1])**2
        if kmax[2] != 0:
            dist += (zvals / kmax[2])**2
        dist = np.sqrt(dist)

        filt = csi_averages * (0.54 + 0.46 * np.cos(np.pi * dist))
        filt[dist > 1] = 0

        return filt[:, :, 0]

    if args.hann_filter:
        filter = make_radial_hann(n2)
        print('Using Hann Filter')
    elif args.csi_filter:
        filter = make_csi_filter(n2, 1)
        print('Using CSI Filter')
    else:
        filter = None
        print('No Filter Selected')


    def process_reference_data(reference_data, n1, n2, num_coils, ft_filter=None):
        center = (n1 - 1) / 2
        max_radius = center  # Can be customized if needed

        # Compute valid phase encode coordinates
        valid_phase_encodes = [
            (j, k) for j in range(n1) for k in range(n1)
            if np.sqrt((j - center) ** 2 + (k - center) ** 2) <= max_radius
        ]
        def grid_to_index(j, k, n1):
            return j * n1 + k
        
        gradient_data = np.transpose(reference_data, (1,2,0,3) ).astype(dtype)
        gradient_data = gradient_data.reshape(len(valid_phase_encodes), num_coils, 50000)
        gradient_data = np.transpose(gradient_data, (0,2,1))

        padded = np.zeros((n1 * n1, 50000, num_coils), dtype=gradient_data.dtype)

        for i, (j, k) in enumerate(valid_phase_encodes):
            idx = grid_to_index(j, k, n1)
            padded[idx] = gradient_data[i]
        reshaped = padded.reshape(n1, n1, 50000, num_coils)

        pad = int((n2-n1)/2)
        pad_width = (
            (pad, pad),      # Axis 0: pad 3 before and 3 after → 7 → 13
            (pad, pad),      # Axis 1: pad 3 before and 3 after → 7 → 13
            (0, 0),      # Axis 2: no padding (50000)
            (0, 0)       # Axis 3: no padding (32 coils)
            )

        reshaped = np.pad(reshaped, pad_width=pad_width, mode='constant', constant_values=0)

        if ft_filter is not None:
            reshaped *= ft_filter[:, :, None, None]

        ft = np.fft.fftshift(np.fft.fft2(reshaped, axes=(0, 1)), axes=(0, 1)).astype(dtype)
        ft_flat = ft.reshape(n2 * n2, 50000, num_coils)

        combined_list = []
        weights_list = []
        for i in range(n2 * n2):
            combined, weights, _ = svd_reduce(ft_flat[i, :, :], return_alpha=True)
            combined_list.append(combined)
            weights_list.append(weights)

        return np.array(combined_list), weights_list

    def process_triangle_data(triangular_data, n1, n2, weights_list, num_coils, ft_filter=None):
        center = (n1 - 1) / 2
        max_radius = center

        # Step 1: Define valid phase encode positions
        valid_phase_encodes = [
            (j, k) for j in range(n1) for k in range(n1)
            if np.sqrt((j - center) ** 2 + (k - center) ** 2) <= max_radius
        ]

        def grid_to_index(j, k, n1):
            return j * n1 + k
        
        gradient_data = np.transpose(triangular_data, (1,2,0,3) ).astype(dtype)
        gradient_data = gradient_data.reshape(len(valid_phase_encodes), num_coils, 50000)
        gradient_data = np.transpose(gradient_data, (0,2,1))

        padded = np.zeros((n1 * n1, 50000, num_coils), dtype=gradient_data.dtype)



        for i, (j, k) in enumerate(valid_phase_encodes):
            idx = grid_to_index(j, k, n1)
            padded[idx] = gradient_data[i]
        reshaped = padded.reshape(n1, n1, 50000, num_coils)

        pad = int((n2-n1)/2)
        pad_width = (
            (pad, pad),      # Axis 0: pad 3 before and 3 after → 7 → 13
            (pad, pad),      # Axis 1: pad 3 before and 3 after → 7 → 13
            (0, 0),      # Axis 2: no padding (50000)
            (0, 0)       # Axis 3: no padding (32 coils)
            )

        reshaped = np.pad(reshaped, pad_width=pad_width, mode='constant', constant_values=0)


        if ft_filter is not None:
            reshaped *= ft_filter[:, :, None, None]

        ft = np.fft.fftshift(np.fft.fft2(reshaped, axes=(0, 1)), axes=(0, 1)).astype(dtype)
        ft_flat = ft.reshape(n2 * n2, 50000, num_coils)

        data_tri_cc = np.empty((n2 * n2, 50000), dtype=dtype)
        for j in range(n2 * n2):
            FID_block = ft_flat[j, :, :]
            weights = weights_list[j]
            combined = weightedCombination(FID_block, weights)
            data_tri_cc[j, :] = combined

        return data_tri_cc

    os.makedirs(output_folder, exist_ok=True)

    destination = output_folder / "InputGradients.npz"
    if Path(args.npz_file).resolve() != destination.resolve():
        shutil.copyfile(args.npz_file, destination)

    dwell_time = 5
    roPts = 50000
    roTime = np.arange(0, dwell_time * roPts, dwell_time)

    for idx in range(4):
        print(f'Processing Slice {idx+1}/4')
        slice_mm = slice_offsets[idx]
        sign = '+' if slice_mm > 0 else '-'
        abs_mm = abs(slice_mm)

        # Lists instead of dicts
        ref_combined_list = []
        tri_plus_cc_list = []
        tri_neg_cc_list = []

        # Loop over 6 batches
        for b in range(6):
            batch_offset = b * batch_size

            # Extract and process reference
            ref_slice = img_obj[:, idx + batch_offset : num_ref + batch_offset : 4, :, :]
            ref_combined_b, weight = process_reference_data(ref_slice, n1, n2, coils, ft_filter=filter)
            ref_combined_list.append(ref_combined_b)

            # Divide triangles into thirds
            for i in range(3):
                tri_start = int(i * num_tri / 3)
                tri_end   = int((i + 1) * num_tri / 3)

                plus_start = idx + num_ref + tri_start + batch_offset
                plus_end   = num_ref + tri_end + batch_offset
                neg_start  = plus_start + 4  # offset by 4 for neg
                neg_end    = plus_end

                tri_plus_slice = img_obj[:, plus_start:plus_end:8, :, :]
                tri_neg_slice  = img_obj[:, neg_start:neg_end:8, :, :]

                tri_plus_cc_list.append(process_triangle_data(tri_plus_slice, n1, n2, weight, coils, ft_filter=filter))
                tri_neg_cc_list.append(process_triangle_data(tri_neg_slice, n1, n2, weight, coils, ft_filter=filter))

        # Stack results
        positive = np.stack(tri_plus_cc_list, axis=0).transpose(2, 1, 0)
        negative = np.stack(tri_neg_cc_list, axis=0).transpose(2, 1, 0)
        ref = np.stack(ref_combined_list, axis=0).transpose(2, 1, 0)

        # Filenames
        ref_filename = f"Ref{sign}{direction}_{int(abs_mm*1000)}_slice.npz"
        pos_filename = f"Positive{sign}{int(abs_mm*1000)}{direction}slice.npz"
        neg_filename = f"Negative{sign}{int(abs_mm*1000)}{direction}slice.npz"

        # Save outputs
        np.savez(
            output_folder / ref_filename,
            acqNum=num_batches,
            avgNum=1,
            dwellTime=dwell_time,
            gradAmp=triangular_amplitudes,
            kspace_all=ref.astype(dtype=dtype),
            roPts=roPts,
            roTime=roTime,
            slice_offset=slice_mm,
            n=n2
        )

        np.savez(
            output_folder / pos_filename,
            acqNum=num_batches,
            avgNum=1,
            dwellTime=dwell_time,
            gradAmp=triangular_amplitudes,
            kspace_all=positive.astype(dtype=dtype),
            roPts=roPts,
            roTime=roTime,
            slice_offset=slice_mm,
            n=n2
        )

        np.savez(
            output_folder / neg_filename,
            acqNum=num_batches,
            avgNum=1,
            dwellTime=dwell_time,
            gradAmp=triangular_amplitudes,
            kspace_all=negative.astype(dtype=dtype),
            roPts=roPts,
            roTime=roTime,
            slice_offset=slice_mm,
            n=n2
        )


if __name__ == "__main__":
    main()