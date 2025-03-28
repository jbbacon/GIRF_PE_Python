import numpy as np
import matplotlib.pyplot as plt

def calculate_output_gradient_optimized(sigS1_POS, sigS1_NEG, sigS2_POS, sigS2_NEG, refS1, refS2, params, Linear = True):
    # Extract the necessary sizes
    nGradAmp = sigS1_POS.shape[1]  # Number of gradient blips
    nRep = sigS1_POS.shape[2]      # Number of repetitions

    # Repeat refS1 and refS2 across repetitions if needed
    refS1 = np.expand_dims(refS1, axis=2)  # Adds a new axis at the 3rd position (8000, 6, 1)
    refS1 = np.tile(refS1, (1, 1, nRep))  # Replicates along the 3rd axis

    refS2 = np.expand_dims(refS2, axis=2)  # Same for refS2
    refS2 = np.tile(refS2, (1, 1, nRep))  # Replicates along the 3rd axis
  
    # Initialize corrected signal arrays
    sigS1Corrected_POS = np.zeros_like(sigS1_POS, dtype=np.complex64)
    sigS1Corrected_NEG = np.zeros_like(sigS1_NEG, dtype=np.complex64)
    sigS2Corrected_POS = np.zeros_like(sigS2_POS, dtype=np.complex64)
    sigS2Corrected_NEG = np.zeros_like(sigS2_NEG, dtype=np.complex64)
    
    # Correct the raw signals by dividing by the corresponding reference
    for nn in range(nGradAmp):
        sigS1Corrected_POS[:, nn, :] = sigS1_POS[:, nn, :] / refS1[:, (nn // 3), :]
        sigS1Corrected_NEG[:, nn, :] = sigS1_NEG[:, nn, :] / refS1[:, (nn // 3), :]
        sigS2Corrected_POS[:, nn, :] = sigS2_POS[:, nn, :] / refS2[:, (nn // 3), :]
        sigS2Corrected_NEG[:, nn, :] = sigS2_NEG[:, nn, :] / refS2[:, (nn // 3), :]

    
    # Compute the temporal derivative of the phase of the signals
    # unwrap is used to handle phase jumps
 
    sigS1Diff_POS = np.diff(np.unwrap(np.angle(sigS1Corrected_POS), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS1Diff_NEG = np.diff(np.unwrap(np.angle(sigS1Corrected_NEG), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS2Diff_POS = np.diff(np.unwrap(np.angle(sigS2Corrected_POS), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)
    sigS2Diff_NEG = np.diff(np.unwrap(np.angle(sigS2Corrected_NEG), axis=0), axis=0) / (params['adcDwellTime'] / 1e6)

    # Compensate for the missing point due to np.diff
    sigS1Diff_POS = np.concatenate((np.zeros((1, sigS1Diff_POS.shape[1], sigS1Diff_POS.shape[2])), sigS1Diff_POS), axis=0)
    sigS1Diff_NEG = np.concatenate((np.zeros((1, sigS1Diff_NEG.shape[1], sigS1Diff_NEG.shape[2])), sigS1Diff_NEG), axis=0)
    sigS2Diff_POS = np.concatenate((np.zeros((1, sigS2Diff_POS.shape[1], sigS2Diff_POS.shape[2])), sigS2Diff_POS), axis=0)
    sigS2Diff_NEG = np.concatenate((np.zeros((1, sigS2Diff_NEG.shape[1], sigS2Diff_NEG.shape[2])), sigS2Diff_NEG), axis=0)

    if Linear == True: 
    # Calculate the output gradient (in rad/s) based on the difference between slices
        gradOutput = ((sigS1Diff_POS - sigS1Diff_NEG)  + (sigS2Diff_NEG - sigS2Diff_POS)) / 4
        gradOutput = gradOutput / (params['gammabar'] * 2 * np.pi) / params['slicePos']  # Convert to mT/m
    elif Linear == False:
        gradOutput = ((sigS1Diff_POS - sigS1Diff_NEG)  - (sigS2Diff_NEG - sigS2Diff_POS)) / 4
        gradOutput = gradOutput / (params['gammabar'] * 2 * np.pi)
    else: 
        print('Error: Select Linear as True or False ')  


    # Convert to single precision (float32)
    gradOutput = gradOutput.astype(np.float32)

    # Compute the FFT of the gradient output to get the frequency domain representation
    gradOutputFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(gradOutput, axes=0), axis=0), axes=0)

    return gradOutput, gradOutputFT
