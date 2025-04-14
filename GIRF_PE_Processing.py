import argparse
import numpy as np
import pandas as pd
import twixtools
from fsl_mrs.utils.preproc.combine import svd_reduce, weightedCombination
import json
import shutil 
from pathlib import Path
import os

# orks out batch information for processing 
def analyze_batches_and_types(csv_filename):
    pulse_sequence_info = pd.read_csv(csv_filename)
    
    batch_indices = pulse_sequence_info['batch_index']
    types = pulse_sequence_info['type']
    
    batch_size = len(batch_indices) // len(batch_indices.unique())
    num_batches = len(batch_indices.unique())
    
    type_counts = pulse_sequence_info.groupby('batch_index')['type'].value_counts().unstack(fill_value=0)
    
    num_ref = type_counts.iloc[0].get('ref', 0)
    num_triangle = type_counts.iloc[0].get('triangle', 0)
    
    return batch_size, num_batches, num_ref, num_triangle


# Generates the pattern from the pulse ordering for coil combination
def generate_batch_pattern(csv_filename):
    pulse_sequence_info = pd.read_csv(csv_filename)
    
    batch_data = pulse_sequence_info[pulse_sequence_info['batch_index'] == 0]
    ref_data = batch_data[batch_data['type'] == 'ref']
    tri_data = batch_data[batch_data['type'] == 'triangle']
    
    ref_mapping = {(row['j'], row['k'], row['slice_offset']): idx for idx, row in ref_data.iterrows()}
    
    pattern = [
        ref_mapping.get((row['j'], row['k'], row['slice_offset'])) 
        for idx, row in tri_data.iterrows()
    ]
    
    if None in pattern:
        raise ValueError("Some triangle slices do not have a corresponding reference slice.")
    
    return pattern


def load_mri_data(filename, start_idx=0, num_scans_to_load=None):
    multi_twix = twixtools.read_twix(filename, parse_data=True)
    end_idx = start_idx + num_scans_to_load
    mdb_data_subset = multi_twix[-1]['mdb'][start_idx:end_idx]
    
    all_data = [mdb.data for mdb in mdb_data_subset if mdb.is_image_scan()]
    
    return np.asarray(all_data)


# Deals with the 80000 points split into 10 groups of 8000
def reshape_mri_data(all_data, batch_size):
    ungrouped = all_data.reshape(batch_size, 10, 32, 8000)
    ungroupedT = np.transpose(ungrouped, (0, 2, 1, 3))
    gradient_data = ungroupedT.reshape(batch_size, 32, 80000)
    return np.transpose(gradient_data, (0, 2, 1))


# Splits reference and gradient data
def process_mri_data(pattern, filename, start_idx=0, num_scans_to_load=None, batch_size=None, num_ref=None):
    all_data = load_mri_data(filename, start_idx, num_scans_to_load)
    data = reshape_mri_data(all_data, batch_size)
    
    data_ref = data[:num_ref]
    data_tri = data[num_ref:]
    
    data_ref_corrected, weightslist = apply_coil_combination(data_ref)
    data_tri_corrected = apply_triangle_weighting(data_tri, pattern, weightslist)
    
    return data_ref_corrected, data_tri_corrected


def apply_coil_combination(data_ref):
    data_ref_corrected = []
    weightslist = []
    
    for i in range(len(data_ref)):
        combined, weights, _ = svd_reduce(data_ref[i], return_alpha=True)
        data_ref_corrected.append(combined)
        weightslist.append(weights)
    
    return np.array(data_ref_corrected), np.array(weightslist)


# Apply reference weighting for the triangular coil combination
def apply_triangle_weighting(data_tri, pattern, weightslist):
    data_tri_corrected = [
        weightedCombination(data_tri[idx], weightslist[element]) 
        for idx, element in enumerate(pattern)
    ]
    return np.array(data_tri_corrected)

# Main Processing
def process_full_mri_data(pattern, filename, batch_size, num_batches, num_ref, num_scans_to_load=None):
    total_scans = batch_size * num_batches * 10
    num_iterations = total_scans // num_scans_to_load
    
    data_ref_corrected_list = []
    data_tri_corrected_list = []
    
    for i in range(num_iterations):
        data_ref_corrected, data_tri_corrected = process_mri_data(
            pattern, filename, start_idx=i * num_scans_to_load, 
            num_scans_to_load=num_scans_to_load, batch_size=batch_size, num_ref=num_ref
        )
        data_ref_corrected_list.append(data_ref_corrected)
        data_tri_corrected_list.append(data_tri_corrected)
    
    final_data_ref_corrected = np.concatenate(data_ref_corrected_list, axis=0)
    final_data_tri_corrected = np.concatenate(data_tri_corrected_list, axis=0)
    
    return final_data_ref_corrected, final_data_tri_corrected


def load_parameters(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'triangular_amplitudes' in data and 'n' in data and 'slice_offset' in data:
        triangular_amplitudes = np.array(data['triangular_amplitudes'])  # assumes list of numbers
        n = data['n']  # assumes scalar
        slice_offset = data['slice_offset']  # could be scalar, list, or array

        return triangular_amplitudes, n, slice_offset
    else:
        raise KeyError("The JSON file does not contain the expected variables: 'triangular_amplitudes', 'n', and 'slice_offset'.")
    

def main(args):
    args.output_folder.mkdir(exist_ok=True, parents=True)
    

    batch_size, num_batches, num_ref, num_triangle = analyze_batches_and_types(args.csv_file)
    pattern = generate_batch_pattern(args.csv_file)

    triangular_amplitudes, n, slice_offset = load_parameters(args.json_file)

    def ref_2dFT(input):
        input_grid = input.reshape(80000, n, n, num_batches)
        ft_kspace = np.fft.fft2(input_grid, axes=(1, 2))
        ft_kspace_shifted = np.fft.fftshift(ft_kspace, axes=(1, 2))
        modified_kspace_all = ft_kspace_shifted.reshape(80000, n*n, num_batches)
        return modified_kspace_all


    def tri_2dFT(input):
        input_grid = input.reshape(80000, n, n, len(triangular_amplitudes))
        ft_kspace = np.fft.fft2(input_grid, axes=(1, 2))
        ft_kspace_shifted = np.fft.fftshift(ft_kspace, axes=(1, 2))
        modified_kspace_all = ft_kspace_shifted.reshape(80000, n*n, len(triangular_amplitudes))
        return modified_kspace_all

    data_ref_corrected, data_tri_corrected = process_full_mri_data(
        pattern, args.mri_file, batch_size, num_batches, num_ref, num_scans_to_load=batch_size * 10
    )

    data_ref = data_ref_corrected.reshape(num_batches, num_ref, 80000)

    dataposy = data_ref[:, 0::2, :].transpose(2, 1, 0)  
    datanegy = data_ref[:, 1::2, :].transpose(2, 1, 0)  

    dataposy_FT = ref_2dFT(dataposy)
    datanegy_FT = ref_2dFT(datanegy)
    
    dwell_time=5
    roPts = 80000
    roTime = np.arange(0, dwell_time * roPts, dwell_time)

    DataRef_pos = {
        'acqNum': num_batches,
        'avgNum': 1,
        'dwellTime': dwell_time,
        'gradAmp': triangular_amplitudes,
        'kspace_all': dataposy_FT,  
        'nch': 1,
        'roPts': roPts,
        'roTime': roTime,
        'slice_offset': slice_offset
    }
    filename_pos = f"{args.output_folder}/Ref+{args.direction}slice.npz"
    np.savez(filename_pos, **DataRef_pos)

    DataRef_neg = {
        'acqNum': num_batches,
        'avgNum': 1,
        'dwellTime': dwell_time,
        'gradAmp': triangular_amplitudes,
        'kspace_all': datanegy_FT,  
        'nch': 1,
        'roPts': roPts,
        'roTime': roTime, 
        'slice_offset': slice_offset
    }
    filename_neg = f"{args.output_folder}/Ref-{args.direction}slice.npz"
    np.savez(filename_neg, **DataRef_neg)

    reshaped_data = data_tri_corrected.reshape(len(triangular_amplitudes), n*n*4, 80000)

    file_paths = [
        f'{args.output_folder}/Positive+{args.direction}slice.npz',
        f'{args.output_folder}/Positive-{args.direction}slice.npz',
        f'{args.output_folder}/Negative+{args.direction}slice.npz',
        f'{args.output_folder}/Negative-{args.direction}slice.npz'
    ]

    for i in range(4):
        selected_data = reshaped_data[:, i::4, :]
        group_data = selected_data.transpose(2, 1, 0)
        group_data_FT = tri_2dFT(group_data)
        
        DataTri = {
            'acqNum': num_triangle,
            'avgNum': 1,
            'dwellTime': dwell_time,
            'gradAmp': triangular_amplitudes,
            'kspace_all': group_data_FT,
            'nch': 1,
            'roPts': roPts,
            'roTime': roTime, 
            'slice_offset': slice_offset
        }
        
        np.savez(file_paths[i], **DataTri)


    input_gradient_file = args.npz_file
    destination = os.path.join(args.output_folder, 'InputGradients.npz')
    if os.path.abspath(input_gradient_file) != os.path.abspath(destination):
        shutil.move(input_gradient_file, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GIRF data")
    parser.add_argument('--mri_file', type=str, required=True, help='Path to the MRI data .dat file')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the pulse sequence CSV file')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the parameters .json file')
    parser.add_argument('--npz_file', type=str, required=True, help='Path to the Input Gradient .npz file')
    parser.add_argument('--output_folder', type=Path, required=True, help='Folder to save the output .npz files')
    parser.add_argument('--direction', type=str, required=True, choices=['x', 'y', 'z'], help='Direction for slice labeling')

    args = parser.parse_args()

    main(args)