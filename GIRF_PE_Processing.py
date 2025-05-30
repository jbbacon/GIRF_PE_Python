#Run as pixi run proc-data --mri_file /path/to/mri_data.dat --csv_file /path/to/pulse_order_log_x.csv --json_file /path/to/parameters.json --npz_file /path/to/InputGradients.npz --direction x --output_folder /path/to/output/folder

import numpy as np
import pandas as pd
import twixtools
from fsl_mrs.utils.preproc.combine import svd_reduce, weightedCombination
import json
import shutil
from pathlib import Path
import os
import argparse

def analyze_batches_and_types(csv_filename):
    pulse_sequence_info = pd.read_csv(csv_filename)
    batch_indices = pulse_sequence_info['batch_index']

    batch_size = len(batch_indices) // len(batch_indices.unique())
    num_batches = len(batch_indices.unique())
    type_counts = pulse_sequence_info.groupby('batch_index')['type'].value_counts().unstack(fill_value=0)
    num_ref = type_counts.iloc[0].get('ref', 0)
    num_triangle = type_counts.iloc[0].get('triangle', 0)
    return batch_size, num_batches, num_ref, num_triangle

def load_mri_data(filename, start_idx=0, num_scans_to_load=None):
    multi_twix = twixtools.read_twix(filename, parse_data=True)
    end_idx = start_idx + num_scans_to_load
    mdb_data_subset = multi_twix[-1]['mdb'][start_idx:end_idx]
    all_data = [mdb.data for mdb in mdb_data_subset if mdb.is_image_scan()]
    return np.asarray(all_data)

def reshape_mri_data(all_data, batch_size, num_coils):
    ungrouped = all_data.reshape(batch_size, 10, num_coils, 8000)
    ungroupedT = np.transpose(ungrouped, (0, 2, 1, 3))
    gradient_data = ungroupedT.reshape(batch_size, num_coils, 80000)
    return np.transpose(gradient_data, (0, 2, 1))

def load_parameters(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if all(k in data for k in ['triangular_amplitudes', 'n', 'slice_offset', 'batch_size']):
        triangular_amplitudes = np.array(data['triangular_amplitudes'])
        n = data['n']
        slice_offset = data['slice_offset']
        batch_size_json = data['batch_size']
        return triangular_amplitudes, n, slice_offset, batch_size_json
    else:
        raise KeyError("JSON missing required keys.")

def process_reference_data(gradient_data, num_ref, n, num_coils):
    def process_ref_block(data_ref_block):
        reshaped = data_ref_block.reshape(n, n, 80000, num_coils)
        ft = np.fft.fftshift(np.fft.fft2(reshaped, axes=(0, 1)), axes=(0, 1))
        ft_flat = ft.reshape(n * n, 80000, num_coils)

        combined_list = []
        weights_list = []
        for i in range(n * n):
            combined, weights, _ = svd_reduce(ft_flat[i, :, :], return_alpha=True)
            combined_list.append(combined)
            weights_list.append(weights)

        return np.array(combined_list), weights_list

    data_ref = gradient_data[:num_ref, :, :]
    data_ref_pos = data_ref[0::2]
    data_ref_neg = data_ref[1::2]

    data_ref_pos_cc, weightslist_pos = process_ref_block(data_ref_pos)
    data_ref_neg_cc, weightslist_neg = process_ref_block(data_ref_neg)

    return data_ref_pos_cc, weightslist_pos, data_ref_neg_cc, weightslist_neg

def process_triangle_data(gradient_data, num_ref, num_triangle, n, weightslist_pos, weightslist_neg, num_coils):
    num_blocks = num_triangle // (4 * n * n)
    data_tri = gradient_data[num_ref:, :, :]
    data_tri_reshape = data_tri.reshape(num_blocks, 4 * n * n, 80000, num_coils)

    directions = {
        0: ("data_tri_0_cc", weightslist_pos),
        1: ("data_tri_1_cc", weightslist_neg),
        2: ("data_tri_2_cc", weightslist_pos),
        3: ("data_tri_3_cc", weightslist_neg)
    }

    data_tri_0_cc = data_tri_1_cc = data_tri_2_cc = data_tri_3_cc = None

    for dir_idx, (var_name, weights_list) in directions.items():
        data_dir = data_tri_reshape[:, dir_idx::4, :, :]
        data_dir_reshape = data_dir.reshape(num_blocks, n, n, 80000, num_coils)
        data_dir_FT = np.fft.fftshift(np.fft.fft2(data_dir_reshape, axes=(1, 2)), axes=(1, 2))
        data_dir_FT = data_dir_FT.reshape(num_blocks, n * n, 80000, num_coils)

        data_dir_cc = np.empty((num_blocks, n * n, 80000), dtype=np.complex128)
        for i in range(num_blocks):
            for j in range(n * n):
                FID_block = data_dir_FT[i, j, :, :]
                weights = weights_list[j]
                combined = weightedCombination(FID_block, weights)
                data_dir_cc[i, j, :] = combined

        if var_name == "data_tri_0_cc":
            data_tri_0_cc = data_dir_cc
        elif var_name == "data_tri_1_cc":
            data_tri_1_cc = data_dir_cc
        elif var_name == "data_tri_2_cc":
            data_tri_2_cc = data_dir_cc
        elif var_name == "data_tri_3_cc":
            data_tri_3_cc = data_dir_cc

    return data_tri_0_cc, data_tri_1_cc, data_tri_2_cc, data_tri_3_cc

def main(args):
    # Step 1: Get batch info
    batch_size, num_batches, num_ref, num_triangle = analyze_batches_and_types(args.csv_file)

    # Step 2: Load parameters
    triangular_amplitudes, n, slice_offset, batch_size_json = load_parameters(args.json_file)

    all_data_ref_pos_cc = []
    all_data_ref_neg_cc = []
    all_data_tri_0_cc = []
    all_data_tri_1_cc = []
    all_data_tri_2_cc = []
    all_data_tri_3_cc = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size * 10
        all_data = load_mri_data(args.mri_file, start_idx=start_idx, num_scans_to_load=batch_size * 10)
        gradient_data = reshape_mri_data(all_data, batch_size, args.coils)

        data_ref_pos_cc, weightslist_pos, data_ref_neg_cc, weightslist_neg = process_reference_data(gradient_data, num_ref, n, args.coils)
        all_data_ref_pos_cc.append(data_ref_pos_cc)
        all_data_ref_neg_cc.append(data_ref_neg_cc)

        tri0, tri1, tri2, tri3 = process_triangle_data(gradient_data, num_ref, num_triangle, n, weightslist_pos, weightslist_neg, args.coils)
        all_data_tri_0_cc.append(tri0)
        all_data_tri_1_cc.append(tri1)
        all_data_tri_2_cc.append(tri2)
        all_data_tri_3_cc.append(tri3)

    all_data_ref_pos_cc = np.stack(all_data_ref_pos_cc, axis=0).transpose(2, 1, 0)
    all_data_ref_neg_cc = np.stack(all_data_ref_neg_cc, axis=0).transpose(2, 1, 0)
    all_data_tri_0_cc = np.concatenate(all_data_tri_0_cc, axis=0).transpose(2, 1, 0)
    all_data_tri_1_cc = np.concatenate(all_data_tri_1_cc, axis=0).transpose(2, 1, 0)
    all_data_tri_2_cc = np.concatenate(all_data_tri_2_cc, axis=0).transpose(2, 1, 0)
    all_data_tri_3_cc = np.concatenate(all_data_tri_3_cc, axis=0).transpose(2, 1, 0)

    os.makedirs(args.output_folder, exist_ok=True)
    dwell_time = 5
    roPts = 80000
    roTime = np.arange(0, dwell_time * roPts, dwell_time)

    np.savez(
        args.output_folder / f"Ref+{args.direction}slice.npz",
        acqNum=num_batches,
        avgNum=1,
        dwellTime=dwell_time,
        gradAmp=triangular_amplitudes,
        kspace_all=all_data_ref_pos_cc.astype(np.complex128),
        nch=1,
        roPts=roPts,
        roTime=roTime,
        slice_offset=slice_offset,
        batch_size=batch_size_json
    )
    np.savez(
        args.output_folder / f"Ref-{args.direction}slice.npz",
        acqNum=num_batches,
        avgNum=1,
        dwellTime=dwell_time,
        gradAmp=triangular_amplitudes,
        kspace_all=all_data_ref_neg_cc.astype(np.complex128),
        nch=1,
        roPts=roPts,
        roTime=roTime,
        slice_offset=slice_offset,
        batch_size=batch_size_json
    )

    tri_list = [
        (all_data_tri_0_cc.astype(np.complex128), f"Positive+{args.direction}slice.npz"),
        (all_data_tri_1_cc.astype(np.complex128), f"Positive-{args.direction}slice.npz"),
        (all_data_tri_2_cc.astype(np.complex128), f"Negative+{args.direction}slice.npz"),
        (all_data_tri_3_cc.astype(np.complex128), f"Negative-{args.direction}slice.npz"),
    ]

    for data, fname in tri_list:
        np.savez(
            args.output_folder / fname,
            acqNum=num_triangle,
            avgNum=1,
            dwellTime=dwell_time,
            gradAmp=triangular_amplitudes,
            kspace_all=data,
            nch=1,
            roPts=roPts,
            roTime=roTime,
            slice_offset=slice_offset,
            batch_size=batch_size_json
        )

    destination = args.output_folder / 'InputGradients.npz'
    if os.path.abspath(args.npz_file) != os.path.abspath(destination):
        shutil.copyfile(args.npz_file, destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GIRF data")
    parser.add_argument('--mri_file', type=str, required=True, help='Path to the MRI data .dat file')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the pulse sequence CSV file')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the parameters .json file')
    parser.add_argument('--npz_file', type=str, required=True, help='Path to the Input Gradient .npz file')
    parser.add_argument('--output_folder', type=Path, required=True, help='Folder to save the output .npz files')
    parser.add_argument('--direction', type=str, required=True, choices=['x', 'y', 'z'], help='Direction for slice labeling')
    parser.add_argument('--coils', type=int, default=32, help='Number of receiver coils (default: 32)')

    args = parser.parse_args()
    main(args)