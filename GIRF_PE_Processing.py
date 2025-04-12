import argparse
import numpy as np
import pandas as pd
import twixtools
from fsl_mrs.utils.preproc import combine
import scipy.io as sio
from scipy.io import savemat, loadmat
from fsl_mrs.utils.preproc.combine import svd_reduce, weightedCombination


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


def reshape_mri_data(all_data, batch_size):
    ungrouped = all_data.reshape(batch_size, 10, 32, 8000)
    ungroupedT = np.transpose(ungrouped, (0, 2, 1, 3))
    gradient_data = ungroupedT.reshape(batch_size, 32, 80000)
    return np.transpose(gradient_data, (0, 2, 1))


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


def apply_triangle_weighting(data_tri, pattern, weightslist):
    data_tri_corrected = [
        weightedCombination(data_tri[idx], weightslist[element]) 
        for idx, element in enumerate(pattern)
    ]
    return np.array(data_tri_corrected)


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


def load_parameters(mat_file):
    data = sio.loadmat(mat_file)
    
    if 'triangular_amplitudes' in data and 'n' in data and 'slice_offset' in data:
        triangular_amplitudes = np.array(data['triangular_amplitudes']).flatten() 
        n = data['n'][0, 0]  
        slice_offset = data['slice_offset']
        return triangular_amplitudes, n, slice_offset
    else:
        raise KeyError("The .mat file does not contain the expected variables: 'triangular_amplitudes' and 'n'.")
    

def main(args):
    batch_size, num_batches, num_ref, num_triangle = analyze_batches_and_types(args.csv_filename)
    pattern = generate_batch_pattern(args.csv_filename)

    triangular_amplitudes, n, slice_offset = load_parameters(args.mat_filename)

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
        pattern, args.mri_filename, batch_size, num_batches, num_ref, num_scans_to_load=batch_size * 10
    )

    data_ref = data_ref_corrected.reshape(num_batches, num_ref, 80000)

    dataposy = data_ref[:, 0::2, :].transpose(2, 1, 0)  
    datanegy = data_ref[:, 1::2, :].transpose(2, 1, 0)  

    dataposy_FT = ref_2dFT(dataposy)
    datanegy_FT = ref_2dFT(datanegy)
    
    dwell_time=5
    roPts = 80000
    roTime = np.arange(0, dwell_time * roPts, dwell_time)

    MatDataRef_pos = {
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
    filename_pos = f"{args.output_folder}/Ref+{args.direction}slice.mat"
    savemat(filename_pos, MatDataRef_pos)

    MatDataRef_neg = {
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
    filename_neg = f"{args.output_folder}/Ref-{args.direction}slice.mat"
    savemat(filename_neg, MatDataRef_neg)

    reshaped_data = data_tri_corrected.reshape(len(triangular_amplitudes), n*n*4, 80000)

    groups = []
    file_paths = [
        f'{args.output_folder}/Positive+{args.direction}slice.mat',
        f'{args.output_folder}/Positive-{args.direction}slice.mat',
        f'{args.output_folder}/Negative+{args.direction}slice.mat',
        f'{args.output_folder}/Negative-{args.direction}slice.mat'
    ]

    for i in range(4):
        selected_data = reshaped_data[:, i::4, :]
        group_data = selected_data.transpose(2, 1, 0)
        group_data_FT = tri_2dFT(group_data)
        
        MatDataTri = {
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
        
        savemat(file_paths[i],  MatDataTri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MRI data")
    
    parser.add_argument('--csv_filename', type=str, required=True, help='Path to the pulse sequence CSV file')
    parser.add_argument('--mri_filename', type=str, required=True, help='Path to the MRI data .dat file')
    parser.add_argument('--mat_filename', type=str, required=True, help='Path to the triangular amplitudes .mat file')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output .mat files')
    parser.add_argument('--direction', type=str, required=True, choices=['x', 'y', 'z'], help='Direction for slice labeling')

    args = parser.parse_args()

    main(args)