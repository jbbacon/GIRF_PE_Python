"""
Bulk work of the calculation done in this file. 
Calculates the spherical harmonics by solving a large series of simultaneous equations for each selected voxel"
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_output_gradient_optimized_spherical(sigS1_POS, sigS1_NEG, sigS2_POS, sigS2_NEG, refS1, refS2, 
                                                      sigS3_POS, sigS3_NEG, sigS4_POS, sigS4_NEG, refS3, refS4,
                                                      params, gradientAxis, order, index1, index2, index3, index4, n2, fov):
    # Extract the necessary sizes
    if gradientAxis == 'x':
        directions = ['x', 'y', 'z']
    if gradientAxis == 'y':
        directions = ['y', 'z', 'x']
    if gradientAxis == 'z':
        directions = ['z', 'x', 'y']

    nGradAmp = sigS1_POS.shape[1]  # Number of gradient blips

    def symmetric_array(n):
        if n % 2 == 0:
            raise ValueError("n must be an odd integer")
        return np.arange(-(n//2), n//2 + 1)

    coords = symmetric_array(n2)* fov/n2

    # Determine valid indices for S1/S2 and S3/S4 early
    valid_indices_s1 = np.array(list(set(index1)))
    valid_indices_s2 = np.array(list(set(index2)))
    valid_indices_s3 = np.array(list(set(index3)))
    valid_indices_s4 = np.array(list(set(index4)))

    # Select only the necessary slices of the data based on valid_indices
    # Ensure indices are within bounds and correctly applied
    sigS1_POS_selected = sigS1_POS[:, :, valid_indices_s1]
    sigS1_NEG_selected = sigS1_NEG[:, :, valid_indices_s1]
    sigS2_POS_selected = sigS2_POS[:, :, valid_indices_s2]
    sigS2_NEG_selected = sigS2_NEG[:, :, valid_indices_s2]
    sigS3_POS_selected = sigS3_POS[:, :, valid_indices_s3]
    sigS3_NEG_selected = sigS3_NEG[:, :, valid_indices_s3]
    sigS4_POS_selected = sigS4_POS[:, :, valid_indices_s4]
    sigS4_NEG_selected = sigS4_NEG[:, :, valid_indices_s4]


    # Repeat refS1, refS2, refS3, and refS4 across repetitions if needed, then select indices
    refS1_selected = np.transpose(np.tile(refS1, (1, 1, 3)), (0,2,1))[:, :, valid_indices_s1]
    refS2_selected = np.transpose(np.tile(refS2, (1, 1, 3)), (0,2,1))[:, :, valid_indices_s2]
    refS3_selected = np.transpose(np.tile(refS3, (1, 1, 3)), (0,2,1))[:, :, valid_indices_s3]
    refS4_selected = np.transpose(np.tile(refS4, (1, 1, 3)), (0,2,1))[:, :, valid_indices_s4]

    # Initialize corrected signal arrays with the selected data's shape
    sigS1Corrected_POS = np.zeros_like(sigS1_POS_selected, dtype=np.complex64)
    sigS1Corrected_NEG = np.zeros_like(sigS1_NEG_selected, dtype=np.complex64)
    sigS2Corrected_POS = np.zeros_like(sigS2_POS_selected, dtype=np.complex64)
    sigS2Corrected_NEG = np.zeros_like(sigS2_NEG_selected, dtype=np.complex64)
    sigS3Corrected_POS = np.zeros_like(sigS3_POS_selected, dtype=np.complex64)
    sigS3Corrected_NEG = np.zeros_like(sigS3_NEG_selected, dtype=np.complex64)
    sigS4Corrected_POS = np.zeros_like(sigS4_POS_selected, dtype=np.complex64)
    sigS4Corrected_NEG = np.zeros_like(sigS4_NEG_selected, dtype=np.complex64)
    
    # Correct the raw signals by dividing by the corresponding reference
    for nn in range(nGradAmp):
        # Division now uses the already selected/sliced reference data
        sigS1Corrected_POS[:, nn, :] = sigS1_POS_selected[:, nn, :] / refS1_selected[:, (nn // 3), :]
        sigS1Corrected_NEG[:, nn, :] = sigS1_NEG_selected[:, nn, :] / refS1_selected[:, (nn // 3), :]
        sigS2Corrected_POS[:, nn, :] = sigS2_POS_selected[:, nn, :] / refS2_selected[:, (nn // 3), :]
        sigS2Corrected_NEG[:, nn, :] = sigS2_NEG_selected[:, nn, :] / refS2_selected[:, (nn // 3), :]
        sigS3Corrected_POS[:, nn, :] = sigS3_POS_selected[:, nn, :] / refS3_selected[:, (nn // 3), :]
        sigS3Corrected_NEG[:, nn, :] = sigS3_NEG_selected[:, nn, :] / refS3_selected[:, (nn // 3), :]
        sigS4Corrected_POS[:, nn, :] = sigS4_POS_selected[:, nn, :] / refS4_selected[:, (nn // 3), :]
        sigS4Corrected_NEG[:, nn, :] = sigS4_NEG_selected[:, nn, :] / refS4_selected[:, (nn // 3), :]
    
    # Compute the temporal derivative of the phase of the signals
    # unwrap is used to handle phase jumps
    sigS1Diff_POS = np.gradient(np.unwrap(np.angle(sigS1Corrected_POS), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS1Diff_NEG = np.gradient(np.unwrap(np.angle(sigS1Corrected_NEG), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS2Diff_POS = np.gradient(np.unwrap(np.angle(sigS2Corrected_POS), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS2Diff_NEG = np.gradient(np.unwrap(np.angle(sigS2Corrected_NEG), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS3Diff_POS = np.gradient(np.unwrap(np.angle(sigS3Corrected_POS), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS3Diff_NEG = np.gradient(np.unwrap(np.angle(sigS3Corrected_NEG), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS4Diff_POS = np.gradient(np.unwrap(np.angle(sigS4Corrected_POS), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS4Diff_NEG = np.gradient(np.unwrap(np.angle(sigS4Corrected_NEG), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)

    sigS1Diff = (sigS1Diff_POS - sigS1Diff_NEG) / 2
    sigS2Diff = (sigS2Diff_POS - sigS2Diff_NEG) / 2
    sigS3Diff = (sigS3Diff_POS - sigS3Diff_NEG) / 2
    sigS4Diff = (sigS4Diff_POS - sigS4Diff_NEG) / 2

    def create_df(x_val, valid_indices):
        rows = []
        # Map original grid_index to its corresponding coordinates
        grid_coords_map = {}
        original_grid_index = 0
        for y_coord in coords:
            for z_coord in coords:
                grid_coords_map[original_grid_index] = {'y': y_coord, 'z': z_coord}
                original_grid_index += 1
        
        for idx in valid_indices:
            # Use the pre-computed coordinates for the valid index
            y_val = grid_coords_map[idx]['y']
            z_val = grid_coords_map[idx]['z']
            rows.append({
                'grid_index': idx, # Keep original grid index for reference if needed
                directions[0]: x_val,
                directions[1]: y_val,
                directions[2]: z_val,
            })
        return pd.DataFrame(rows)

    sigS1Diff_df = create_df(params['slicePos1'], valid_indices_s1)
    sigS2Diff_df = create_df(params['slicePos2'], valid_indices_s2)
    sigS3Diff_df = create_df(params['slicePos3'], valid_indices_s3)
    sigS4Diff_df = create_df(params['slicePos4'], valid_indices_s4)

    # Combine DataFrames
    combined_df = pd.concat([sigS1Diff_df, sigS2Diff_df, sigS3Diff_df, sigS4Diff_df], ignore_index=True)

    def solid_harmonic_basis(x, y, z, s_order):
        r2 = x**2 + y**2 + z**2
        basis = []

        if s_order >= 0:
            basis.extend([1])
        if s_order >= 1:
            basis.extend([x, y, z])
        if s_order >= 2:
            basis.extend([x * y, z * y, 3 * z**2 - r2, x * z, x**2 - y**2])
        if s_order >= 3:
            basis.extend([
                y * (3 * x**2 - y**2),
                x * y * z,
                y * (5 * z**2 - r2),
                5 * z**3 - 3 * z * r2,
                x * (5 * z**2 - r2),
                (x**2 - y**2) * z,
                x * (x**2 - 3 * y**2)
            ])
        return np.array(basis)

    def build_design_matrix(df):
        rows = []
        for _, row in df.iterrows():
            x, y, z = row['x'], row['y'], row['z']
            rows.append(solid_harmonic_basis(x, y, z, order))
        return np.array(rows)

    # Build design matrix A and pseudoinverse
    A = build_design_matrix(combined_df)
    A_pinv = np.linalg.pinv(A)

    # The signals `sigS1Diff`, `sigS2Diff`, etc. are already trimmed due to earlier selection
    # So we can directly concatenate them.
    s_concat_all = np.concatenate([sigS1Diff, sigS2Diff, sigS3Diff, sigS4Diff], axis=-1)

    assert s_concat_all.shape[2] == A.shape[0]

    # Solve for coefficients
    coeffs_all = np.einsum('nrm,mb->nrb', s_concat_all, A_pinv.T)
    coeffs_all /= (params['gammabar'] * 2 * np.pi)

    # Fourier transform along the first axis (time)
    coeffs_all_FT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(coeffs_all, axes=0), axis=0), axes=0)

    return coeffs_all, coeffs_all_FT