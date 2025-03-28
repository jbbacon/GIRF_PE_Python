import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def read_multi_files(var_name, *file_paths, N, F):
    """
    Function for loading variables with the same name from multiple files.

    Parameters:
    var_name (str): The name of the variable to load from each file.
    file_paths (str): Paths to the files to be loaded.

    Returns:
    list: A list of variables containing the loaded data from the corresponding files.

    Example:
    To load three T2* decays from three files with variable name 'signal':
    [S1, S2, S3] = read_multi_files('signal', 'FileName1', 'FileName2', 'FileName3')
    """
    
    if len(file_paths) == 0:
        raise ValueError("At least one file name/path is needed.")

    if not isinstance(var_name, str):
        raise TypeError("The first input argument should be a string for the variable name.")

    if not all(isinstance(file, str) for file in file_paths):
        raise TypeError("The file paths should be strings.")

    # Initialize the first file to get the general structure
    file_name = file_paths[0]
    data = sio.loadmat(file_name)

    # Extracting the variable to check consistency
    first_var = data[var_name]

    first_var = first_var[N:F, ...]

    # Retrieve the metadata from the first file
    ro_pts = data['roPts']
    nch = data['nch']
    dwell_time = data['dwellTime']
    xpts = first_var.shape[0]

    # Initialize the output list with the first variable
    result = [first_var]

    # Iterate over the remaining files to check consistency and load data
    for file_name in file_paths[1:]:
        data = sio.loadmat(file_name)
        

        result.append(data[var_name][N:F, ...])

    return result
