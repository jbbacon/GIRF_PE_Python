import numpy as np

def read_multi_files(var_name, *file_paths, N, F):
    """
    Function for loading variables with the same name from multiple .npz files.

    Parameters:
    var_name (str): The name of the variable to load from each file.
    file_paths (str): Paths to the .npz files to be loaded.
    N (int): Start index for slicing the data.
    F (int): End index for slicing the data.

    Returns:
    list: A list of variables containing the sliced data from the corresponding files.

    Example:
    To load three T2* decays from three files with variable name 'signal':
    [S1, S2, S3] = read_multi_files('signal', 'File1', 'File2', 'File3.', N=0, F=30000)
    """
    
    if len(file_paths) == 0:
        raise ValueError("At least one file name/path is needed.")

    if not isinstance(var_name, str):
        raise TypeError("The first input argument should be a string for the variable name.")

    if not all(isinstance(file, str) for file in file_paths):
        raise TypeError("The file paths should be strings.")

    result = []
    for file_name in file_paths:
        with np.load(file_name) as data:
            if var_name not in data:
                raise KeyError(f"Variable '{var_name}' not found in file: {file_name}")
            result.append(data[var_name][N:F, ...])

    return result
